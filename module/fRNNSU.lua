--[[

 flexible Recurrent Neural Network.

--]]

local RNNSU, parent = torch.class('vqalstm.RNNSU', 'vqalstm.RNN')

function RNNSU:__init(config)
  parent.__init(self, config)
end

-- Instantiate a new RNNSU cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function RNNSU:new_cell()
  local input = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer RNNSU
  local htable = {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)

    local new_layer = function()
      local in_module = (layer == 1)
        and nn.Linear(self.in_dim, self.mem_dim)(input)
        or  nn.Linear(self.mem_dim, self.mem_dim)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        h_p
      }
    end

    -- update the state of the RNNSU cell
    htable[layer] = nn.Tanh()(new_layer())
  end

  -- if RNNSU is single-layered, this makes htable/ctable Tensors (instead of tables).
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
-- reverse: if true, read the input from right to left (useful for bidirectional RNNs).
-- Returns the final hidden state of the RNNSU.
function RNNSU:forward(inputs, reverse)
  return parent.forward(self, inputs, reverse)
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function RNNSU:backward(inputs, grad_outputs, reverse)
  return parent.backward(self, inputs, grad_outputs, reverse)
end

function RNNSU:share(lstm, ...)
  parent.share(self, lstm, ...)
end

function RNNSU:zeroGradParameters()
  parent.zeroGradParameters(self)
end

function RNNSU:parameters()
  return parent.parameters(self)
end

-- Clear saved gradients
function RNNSU:forget()
  parent.forget(self)
end
