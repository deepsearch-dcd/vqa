-- LSTM modified from http://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua
-- ref: char-rnn
--[[
  LSTM = require 'modules/LSTM'

  -- single rnn has 3 inputs, 2 outputs
  singlernn = LSTM.single(input_size, rnn_size)
  inputs = {torch.rand(input_size), torch.rand(rnn_size), torch.rand(rnn_size)}
  outputs = singlernn:forward(inputs)

  -- multiple rnn has (2*num_layer+1) inputs, (2*num_layer) outputs
  rnn_sizes = {rnn_size1, rnn_size2}
  multiplernn = LSTM.multiple(input_size, rnn_sizes, num_layer[, dropout])
  inputs = {torch.rand(input_size), torch.rand(rnn_size1), torch.rand(rnn_size1),
            torch.rand(rnn_size2), torch.rand(rnn_size2)}
  outputs = multiplernn:forward(inputs)

  -- general rnn
  rnnmodule1 = LSTM.create(input_size, rnn_size)
  rnnmodule2 = LSTM.create(input_size, rnn_size, num_layer[, dropout])
  rnnmodule3 = LSTM.create(input_size, rnn_sizes, num_layer[, dropout])
--]]
require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.single(input_size, rnn_size)
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  table.insert(inputs, nn.Identity()())   -- c at time t-1
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  local input = inputs[1]
  local prev_c = inputs[2]
  local prev_h = inputs[3]

  --------------------- preactivations ----------------------
  local i2h = nn.Linear(input_size, 4 * rnn_size)(input)     -- input to hidden
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)      -- hidden to hidden
  local preactivations = nn.CAddTable()({i2h, h2h})          -- i2h + h2h

  ------------------ non-linear transforms ------------------
  -- gates
  local gates_chunk = nn.Narrow(1, 1, 3 * rnn_size)(preactivations)
  local all_gates = nn.Sigmoid()(gates_chunk)

  -- input
  local in_chunk = nn.Narrow(1, 3 * rnn_size + 1, rnn_size)(preactivations)
  local in_transform = nn.Tanh()(in_chunk)

  ---------------------- gate narrows -----------------------
  local in_gate = nn.Narrow(1, 1, rnn_size)(all_gates)
  local forget_gate = nn.Narrow(1, rnn_size + 1, rnn_size)(all_gates)
  local out_gate = nn.Narrow(1, 2 * rnn_size + 1, rnn_size)(all_gates)

  --------------------- next cell state ---------------------
  local c_forget = nn.CMulTable()({forget_gate, prev_c})  -- previous cell state contribution
  local c_input = nn.CMulTable()({in_gate, in_transform}) -- input contribution
  local next_c = nn.CAddTable()({c_forget, c_input})

  -------------------- next hidden state --------------------
  local c_transform = nn.Tanh()(next_c)
  local next_h = nn.CMulTable()({out_gate, c_transform})

  --------------------- output structure --------------------
  local outputs = {}
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

function LSTM.multiple(input_size, rnn_sizes, num_layer, dropout)
  -- rnn_sizes has num_layer elements
  dropout = dropout or 0
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  for L = 1,num_layer do
    table.insert(inputs, nn.Identity()())   -- c at time t-1
    table.insert(inputs, nn.Identity()())   -- h at time t-1
  end

  local inputs_L = {}, input_size_L
  local outputs = {}
  -- for layer 1
  table.insert(inputs_L, inputs[1])
  input_size_L = input_size
  for L = 1,num_layer do
    table.insert(inputs_L, inputs[L*2])
    table.insert(inputs_L, inputs[L*2+1])
    local rnn_size_L = rnn_sizes[L]

    local outputs_L = LSTM.single(input_size_L, rnn_size_L)(inputs_L)
    local next_c = nn.SelectTable(1)(outputs_L)
    local next_h = nn.SelectTable(2)(outputs_L)

    --------------------- output structure --------------------
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)

    -- for next layer
    inputs_L = {}
    local next_input = next_h
    if dropout > 0 then next_input = nn.Dropout(dropout)(next_input) end
    table.insert(inputs_L, next_input)
    input_size_L = rnn_size_L
  end

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

function LSTM.create(input_size, rnn_size, num_layer, dropout)
  if type(rnn_size) == 'number' then -- rnn_size contain a value
    if num_layer then -- num_layer is not nil (multiple layers with equal rnn size)
      assert(num_layer > 0, 'num_layer MUST be GREATER THAN 0!')
      local rnn_sizes = {}
      for i=1,num_layer do table.insert(rnn_sizes, rnn_size) end
      return LSTM.multiple(input_size, rnn_sizes, num_layer, dropout)
    else -- num_layer is nil (one layer rnn)
      return LSTM.single(input_size, rnn_size)
    end
  else -- rnn_size contain a table of values (multiple layers with different rnn size)
    assert(#rnn_size == num_layer, 'number of rnn_sizes MUST EQUAL num_layer!')
    return LSTM.multiple(input_size, rnn_size, num_layer, dropout)
  end
end

return LSTM