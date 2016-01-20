require 'nn'
require 'nngraph'
require 'module/sentenceCNN'

local question = nn.Identity()()
local emb_question = nn.LookupTable(857, 50)(question)
local se = sentenceCNN(50, {{3,1,200,2,2},
                    {3,1,300,2,2},
                    {3,1,300,2,2}})(emb_question)
local prob = nn.LogSoftMax()(nn.Linear(2*300, 969)(nn.Reshape(600)(se)))

return nn.gModule({question}, {prob})
