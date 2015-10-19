require 'nn'
require 'nngraph'
local RNN = require 'module/RNN'

local inputs
local outputs

local input_size = 30
local rnn_size = 20
local rnn_size1 = 20
local rnn_size2 = 10
local rnn_size3 = 5
local rnn_sizes = {rnn_size1, rnn_size2, rnn_size3}
local num_layer = #rnn_sizes
local dropout = 0.5

local input = torch.rand(input_size)
local prev_h = torch.rand(rnn_size)
local prev_h1 = torch.rand(rnn_size1)
local prev_h2 = torch.rand(rnn_size2)
local prev_h3 = torch.rand(rnn_size3)

-- single rnn
local singlernn = RNN.single(input_size, rnn_size)
inputs = {input, prev_h}
outputs = singlernn:forward(inputs)
local next_h = outputs
print('input size: ' .. input:size()[1])
print('rnn size: ' .. prev_h:size()[1])
print('next_h size: ' .. next_h:size()[1])
print('single outputs:', outputs)

-- multiple rnn
local multiplernn = RNN.multiple(input_size, rnn_sizes, num_layer, dropout)
inputs = {input, prev_h1, prev_h2, prev_h3}
outputs = multiplernn:forward(inputs)
print('multiple outputs:', outputs)

-- general rnn modules
local rnnmodule1 = RNN.create(input_size, rnn_size)
inputs = {input, prev_h}
outputs = rnnmodule1:forward(inputs)
print('general single outputs:', outputs)
local rnnmodule2 = RNN.create(input_size, rnn_size, num_layer, dropout)
inputs = {input, prev_h, prev_h, prev_h}
outputs = rnnmodule2:forward(inputs)
print('general multiple same sized rnn outputs:', outputs)
local rnnmodule3 = RNN.create(input_size, rnn_sizes, num_layer, dropout)
local rnnmodule4 = RNN.create(input_size, rnn_sizes, num_layer)
inputs = {input, prev_h1, prev_h2, prev_h3}
outputs = rnnmodule3:forward(inputs)
print('general multiple outputs:', outputs)
outputs = rnnmodule4:forward(inputs)
print('general multiple no dropout outputs:', outputs)
