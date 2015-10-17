-- LSTM modified from http://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua
-- ref: char-rnn
--[[
  local LSTM = require 'modules/LSTM.lua'
  rnnmodule = LSTM.create(input_size, rnn_size, num_layer[, dropout])
  (2*num_layer+1) inputs, (2*num_layer) outputs
--]]
require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, rnn_size, num_layer, dropout)
  dropout = dropout or 0
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  for L = 1,num_layer do
    table.insert(inputs, nn.Identity()())   -- c at time t-1
    table.insert(inputs, nn.Identity()())   -- h at time t-1
  end

  local input_L, input_size_L
  local outputs = {}
  for L = 1,num_layer do
    if L == 1 then
      input_L = inputs[1]
      input_size_L = input_size
    else
      input_L = outputs[(L-1)*2]
      if dropout > 0 then input_L = nn.Dropout(dropout)(input_L) end
      input_size_L = rnn_size
    end
    local prev_c = inputs[L*2]
    local prev_h = inputs[L*2+1]

    --------------------- preactivations ----------------------
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(input_L) -- input to hidden
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
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

return LSTM