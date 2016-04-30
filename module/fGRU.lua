--[[

 flexible Gated Recurrent Unit.
 treelstm -> vqalstm
--]]

local GRU, parent = torch.class('vqalstm.GRU', 'nn.Module')

function GRU:__init(config)
  parent.__init(self)

  self.in_dim = config.in_dim
  self.mem_dim = config.mem_dim or 150
  self.num_layers = config.num_layers or 1
  self.cuda = config.cuda or false

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {}  -- table of cells in a roll-out

  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local htable_init, htable_grad
  if self.num_layers == 1 then
    htable_init = torch.zeros(self.mem_dim)
    htable_grad = torch.zeros(self.mem_dim)
    if self.cuda then
      htable_init = htable_init:float():cuda()
      htable_grad = htable_grad:float():cuda()
    end
  else
    htable_init, htable_grad = {}, {}
    for i = 1, self.num_layers do
      htable_init[i] = torch.zeros(self.mem_dim)
      htable_grad[i] = torch.zeros(self.mem_dim)
      if self.cuda then
        htable_init[i] = htable_init[i]:float():cuda()
        htable_grad[i] = htable_grad[i]:float():cuda()
      end
    end
  end
  self.initial_values = htable_init
  self.gradInput = {
    torch.zeros(self.in_dim),
    htable_grad
  }
  if self.cuda then
    self.gradInput[1] = self.gradInput[1]:float():cuda()
  end
end

-- Instantiate a new GRU cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function GRU:new_cell()
  local input = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer GRU
  local htable = {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)

    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(self.in_dim, self.mem_dim)(input)
        or  nn.Linear(self.mem_dim, self.mem_dim)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        nn.Linear(self.mem_dim, self.mem_dim)(h_p)
      }
    end

    -- update and reset gates
    local u = nn.Sigmoid()(new_gate()) -- update_gate
    local r = nn.Sigmoid()(new_gate()) -- reset_gate
    local gated_h = nn.CMulTable()({r, h_p}) -- gated_hidden
    local h_candidate = nn.Tanh()(nn.CAddTable(){ -- hidden_candidate
      (layer == 1) and nn.Linear(self.in_dim, self.mem_dim)(input)
      or nn.Linear(self.mem_dim, self.mem_dim)(htable[layer - 1]),
      nn.Linear(self.mem_dim, self.mem_dim)(gated_h)
    })

    -- update the state of the GRU hidden state
    local zh = nn.CMulTable()({u, h_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(u)), h_p})

    htable[layer] = nn.CAddTable()({zh, zhm1})
  end

  -- if GRU is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable = nn.Identity()(htable)
  local cell = nn.gModule({input, htable_p}, {htable})
  if self.cuda then
    cell = cell:cuda()
  end

  -- share parameters
  if self.master_cell then
    share_params(cell, self.master_cell)
  end
  return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional GRUs).
-- Returns the final hidden state of the GRU.
function GRU:forward(inputs, reverse)
  local size = inputs:size(1)
  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end

    local outputs = cell:forward({input, prev_output})
    local htable = outputs
    if self.num_layers == 1 then
      self.output = htable
    else
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = htable[i]
      end
    end
  end
  return self.output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function GRU:backward(inputs, grad_outputs, reverse)
  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end

  local input_grads = torch.Tensor(inputs:size())
  if self.cuda then
    input_grads = input_grads:float():cuda()
  end
  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = self.gradInput[2]
    if self.num_layers == 1 then
      grads:add(grad_output)
    else
      for i = 1, self.num_layers do
        grads[i]:add(grad_output[i])
      end
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or self.initial_values
    self.gradInput = cell:backward({input, prev_output}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return input_grads
end

function GRU:share(gru, ...)
  if self.in_dim ~= gru.in_dim then error("GRU input dimension mismatch") end
  if self.mem_dim ~= gru.mem_dim then error("GRU memory dimension mismatch") end
  if self.num_layers ~= gru.num_layers then error("GRU layer count mismatch") end
  share_params(self.master_cell, gru.master_cell, ...)
end

function GRU:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function GRU:parameters()
  return self.master_cell:parameters()
end

-- Clear saved gradients
function GRU:forget()
  self.depth = 0
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end
