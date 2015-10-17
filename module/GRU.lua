-- GRU modified from http://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua
-- ref: char-rnn
--[[
  local GRU = require 'modules/GRU.lua'
  rnnmodule = GRU.create(input_size, rnn_size, num_layer[, dropout])
  (num_layer+1) inputs, num_layer outputs
--]]
require 'nn'
require 'nngraph'

local GRU = {}

function GRU.create(input_size, rnn_size, num_layer, dropout)
  dropout = dropout or 0
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  for L = 1,num_layer do
    table.insert(inputs, nn.Identity()())   -- h at time t-1
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local input_L, input_size_L
  local outputs = {}
  for L = 1,num_layer do
    if L == 1 then
      input_L = inputs[1]
      input_size_L = input_size
    else
      input_L = outputs[(L-1)]
      if dropout > 0 then input_L = nn.Dropout(dropout)(input_L) end
      input_size_L = rnn_size
    end
    local prev_h = inputs[L+1]

    ------------------------- GRU tick ------------------------
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, input_L, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, input_L, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local hidden_candidate = nn.Tanh()(new_input_sum(input_size_L,input_L,gated_hidden))

    -------------------- next hidden state --------------------
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    --------------------- output structure --------------------
    table.insert(outputs, next_h)
  end

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

return GRU