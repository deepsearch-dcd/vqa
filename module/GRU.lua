-- GRU modified from http://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua
-- ref: char-rnn
--[[
  GRU = require 'modules/GRU'

  -- single rnn has 2 inputs, 1 outputs
  singlernn = GRU.single(input_size, rnn_size)
  inputs = {torch.rand(input_size), torch.rand(rnn_size)}
  outputs = singlernn:forward(inputs)

  -- multiple rnn has (num_layer+1) inputs, (num_layer) outputs
  rnn_sizes = {rnn_size1, rnn_size2}
  multiplernn = GRU.multiple(input_size, rnn_sizes, num_layer[, dropout])
  inputs = {torch.rand(input_size), torch.rand(rnn_size1), torch.rand(rnn_size2)}
  outputs = multiplernn:forward(inputs)

  -- general rnn
  rnnmodule1 = GRU.create(input_size, rnn_size)
  rnnmodule2 = GRU.create(input_size, rnn_size, num_layer[, dropout])
  rnnmodule3 = GRU.create(input_size, rnn_sizes, num_layer[, dropout])
--]]
require 'nn'
require 'nngraph'

local GRU = {}

function GRU.single(input_size, rnn_size)
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  local input = inputs[1]
  local prev_h = inputs[2]

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  ------------------------- GRU tick ------------------------
  -- forward the update and reset gates
  local update_gate = nn.Sigmoid()(new_input_sum(input_size, input, prev_h))
  local reset_gate = nn.Sigmoid()(new_input_sum(input_size, input, prev_h))
  -- compute candidate hidden state
  local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
  local hidden_candidate = nn.Tanh()(new_input_sum(input_size,input,gated_hidden))

  -------------------- next hidden state --------------------
  -- compute new interpolated hidden state, based on the update gate
  local zh = nn.CMulTable()({update_gate, hidden_candidate})
  local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
  local next_h = nn.CAddTable()({zh, zhm1})

  --------------------- output structure --------------------
  local outputs = {}
  table.insert(outputs, next_h)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

function GRU.multiple(input_size, rnn_sizes, num_layer, dropout)
  dropout = dropout or 0
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  for L = 1,num_layer do
    table.insert(inputs, nn.Identity()())   -- h at time t-1
  end

  local inputs_L = {}, input_size_L
  local outputs = {}
  -- for layer 1
  table.insert(inputs_L, inputs[1])
  input_size_L = input_size
  for L = 1,num_layer do
    table.insert(inputs_L, inputs[L+1]) -- prev_h
    local rnn_size_L = rnn_sizes[L]

    local outputs_L = GRU.single(input_size_L, rnn_size_L)(inputs_L)
    local next_h = (outputs_L)

    --------------------- output structure --------------------
    table.insert(outputs, next_h)

    -- for next layer
    inputs_L = {}
    local next_input = next_h
    if dropout > 0 then next_input = nn.Dropout(dropout)(next_input) end
    table.insert(inputs_L, next_input) -- next_h
    input_size_L = rnn_size_L
  end

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

function GRU.create(input_size, rnn_size, num_layer, dropout)
  if type(rnn_size) == 'number' then -- rnn_size contain a value
    if num_layer then -- num_layer is not nil (multiple layers with equal rnn size)
      assert(num_layer > 0, 'num_layer MUST be GREATER THAN 0!')
      local rnn_sizes = {}
      for i=1,num_layer do table.insert(rnn_sizes, rnn_size) end
      return GRU.multiple(input_size, rnn_sizes, num_layer, dropout)
    else -- num_layer is nil (one layer rnn)
      return GRU.single(input_size, rnn_size)
    end
  else -- rnn_size contain a table of values (multiple layers with different rnn size)
    assert(#rnn_size == num_layer, 'number of rnn_sizes MUST EQUAL num_layer!')
    return GRU.multiple(input_size, rnn_size, num_layer, dropout)
  end
end

return GRU