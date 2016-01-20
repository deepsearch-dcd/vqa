require 'nn'
require 'nngraph'


local input = nn.Identity()() -- assume dims as 18 x 150

-- compute 18 choice parallelly
local prl = nn.ParallelTable()
local mlp = nn.Sequential()
mlp:add(nn.Linear(150, 100))
mlp:add(nn.Linear(100, 50))
mlp:add(nn.Linear(50, 1))
prl:add(mlp)
for i = 2, 18 do
    --prl:add(mlp:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    prl:add(mlp:clone('weight', 'bias'))
end

-- compute the gap between true and false choice
local scores = prl(nn.SplitTable(1,2)(input))
local pos = nn.SelectTable(1)(scores)
local neg = nn.Max(1, 1)(nn.JoinTable(1,1)(nn.NarrowTable(2, 17)(scores)))
local gap = nn.CSubTable()({pos, neg})

return nn.gModule({input}, {gap})

