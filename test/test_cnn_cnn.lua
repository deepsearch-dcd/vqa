require 'nn'
require 'cunn'
require 'util/SGDTrainer'

local util = require 'util/util'
local model = require 'model/cnn_cnn'

print 'test cnn_cnn...'
local dataset = {}
dataset.questions = (torch.rand(10, 30)*857):int()
dataset.images = torch.rand(10, 1000)
dataset.answers = (torch.rand(10)*969):int()
setmetatable(dataset, { __index = 
	function(t, i)
		return {{t.images[i], t.questions[i]}, t.answers[i]}
	end}
)
function dataset:size() return self.images:size(1) end
util.to_cuda(dataset)

model = model:cuda()
local criterion = nn.ClassNLLCriterion():cuda()
local trainer = SGDTrainer(model, criterion)
trainer.maxEpoch = 10
trainer.snapshotIter = nil
trainer.verbose = false

trainer:train(dataset)
print 'OK'
