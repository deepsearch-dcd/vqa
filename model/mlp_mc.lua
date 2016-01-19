require 'nn'

local mlp = nn.Sequential()
mlp:add(nn.Linear(150, 100))
mlp:add(nn.Linear(100, 50))
mlp:add(nn.Linear(50, 1))

return mlp
