require 'nn'
require 'nngraph'
require 'module/sentenceCNN'

local question = nn.Identity()()
local image = nn.Identity()()
local se = sentenceCNN(50, {{3,1,200,2,2},
                            {3,1,300,2,2},
                            {3,1,300,2,2}})(question)
local word1, word2 = nn.SplitTable(1, 2)(se):split(2)
local iword = nn.Linear(1000,300)(image)
local mword = nn.JoinTable(1, 1)({word1, iword, word2})
local prob = nn.LogSoftMax()(nn.Linear(400, 969)(nn.Linear(900, 400)(mword)))

return nn.gModule({image, question}, {prob})
