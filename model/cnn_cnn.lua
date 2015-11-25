require 'nn'
require 'nngraph'
require 'module/sentenceCNN'

local question = nn.Identity()()
local image = nn.Identity()()
local emb_question = nn.LookupTable(857, 50)(question)
local se = sentenceCNN(50, {{3,1,200,2,2},
			    {3,1,300,2,2},
			    {3,1,300,2,2}})(emb_question)
local word1, word2 = nn.SplitTable(1, 2)(se):split(2)
local iword = nn.Linear(1000,300)(image)
local mword = nn.JoinTable(1, 1)({word1, iword, word2})
local prob = nn.LogSoftMax()(nn.Linear(3*300, 969)(mword))

return nn.gModule({image, question},{prob})
