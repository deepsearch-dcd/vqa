-- RNN modified from http://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua
-- ref: char-rnn
--[[
  RNN = require 'modules/RNN'

  -- single rnn has 2 inputs, 1 outputs
  singlernn = RNN.single(input_size, rnn_size)
  inputs = {torch.rand(input_size), torch.rand(rnn_size)}
  outputs = singlernn:forward(inputs)

  -- multiple rnn has (num_layer+1) inputs, (num_layer) outputs
  rnn_sizes = {rnn_size1, rnn_size2}
  multiplernn = RNN.multiple(input_size, rnn_sizes, num_layer[, dropout])
  inputs = {torch.rand(input_size), torch.rand(rnn_size1), torch.rand(rnn_size2)}
  outputs = multiplernn:forward(inputs)

  -- general rnn
  rnnmodule1 = RNN.create(input_size, rnn_size)
  rnnmodule2 = RNN.create(input_size, rnn_size, num_layer[, dropout])
  rnnmodule3 = RNN.create(input_size, rnn_sizes, num_layer[, dropout])
--]]
require 'nn'
require 'nngraph'

local RNN = {}

function RNN.single(input_size, rnn_size)
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  local input = inputs[1]
  local prev_h = inputs[2]

  -------------------- next hidden state --------------------
  local i2h = nn.Linear(input_size, rnn_size)(input) -- input to hidden
  local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)  -- hidden to hidden
  local next_h = nn.CAddTable()({i2h, h2h})          -- i2h + h2h

  --------------------- output structure --------------------
  local outputs = {}
  table.insert(outputs, next_h)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

function RNN.multiple(input_size, rnn_sizes, num_layer, dropout)
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

    local outputs_L = RNN.single(input_size_L, rnn_size_L)(inputs_L)
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

function RNN.create(input_size, rnn_size, num_layer, dropout)
  if type(rnn_size) == 'number' then -- rnn_size contain a value
    if num_layer then -- num_layer is not nil (multiple layers with equal rnn size)
      assert(num_layer > 0, 'num_layer MUST be GREATER THAN 0!')
      local rnn_sizes = {}
      for i=1,num_layer do table.insert(rnn_sizes, rnn_size) end
      return RNN.multiple(input_size, rnn_sizes, num_layer, dropout)
    else -- num_layer is nil (one layer rnn)
      return RNN.single(input_size, rnn_size)
    end
  else -- rnn_size contain a table of values (multiple layers with different rnn size)
    assert(#rnn_size == num_layer, 'number of rnn_sizes MUST EQUAL num_layer!')
    return RNN.multiple(input_size, rnn_size, num_layer, dropout)
  end
end

return RNN