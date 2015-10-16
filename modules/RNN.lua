-- RNN modified from http://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua
-- ref: char-rnn
--[[
  local RNN = require 'modules/RNN.lua'
  rnnmodule = RNN.create(input_size, rnn_size, num_layer[, dropout])
  (num_layer+1) inputs, num_layer outputs
--]]
require 'nn'
require 'nngraph'

local RNN = {}

function RNN.create(input_size, rnn_size, num_layer, dropout)
  dropout = dropout or 0
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  for L = 1,num_layer do
    table.insert(inputs, nn.Identity()())   -- h at time t-1
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

    -------------------- next hidden state --------------------
    local i2h = nn.Linear(input_size, rnn_size)(input_L) -- input to hidden
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)    -- hidden to hidden
    local next_h = nn.CAddTable()({i2h, h2h})            -- i2h + h2h

    --------------------- output structure --------------------
    table.insert(outputs, next_h)
  end

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

return RNN