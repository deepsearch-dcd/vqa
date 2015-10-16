require 'nn'
require 'nngraph'
local LSTM = require 'modules/LSTM.lua'

local input_size = 30
local rnn_size = 20
local num_layer = 2
local dropout = 0.5
local x = torch.rand(input_size)
local inputs = {x, torch.rand(rnn_size), torch.rand(rnn_size), torch.rand(rnn_size), torch.rand(rnn_size)}

local rnnmodule = LSTM.create(input_size, rnn_size, num_layer, dropout)
output = rnnmodule:forward(inputs)
