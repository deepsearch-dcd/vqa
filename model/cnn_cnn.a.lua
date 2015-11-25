require 'nn'
require 'nngraph'
require 'module/sentenceCNN'

local question = nn.Identity()()
local image = nn.Identity()()
local se = sentenceCNN(50, {{3,1,200,2,2},
                            {3,1,300,2,2},
                            {3,1,300,2,2}})(question)
local word1, word2 = nn.SplitTable(1, 2)(se):split(2)
word1 = nn.Reshape(1,300)(word1)
word2 = nn.Reshape(1,300)(word2)
local iword = nn.Reshape(1,300)(nn.Linear(1000,300)(image))
local mword = nn.TemporalConvolution(300, 400, 3)(nn.JoinTable(1, 2)({word1, iword, word2}))
local prob = nn.LogSoftMax()(nn.Linear(400, 969)(nn.View(400):setNumInputDims(2)(mword)))

return nn.gModule({image, question}, {prob})
