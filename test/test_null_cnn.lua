require 'nn'
require 'cunn'
require 'util/train'

local util = require 'util/util'
local model = require 'model/null_cnn'

print 'test null_cnn...'
local dataset = {}
dataset.questions = (torch.rand(10,30)*857+1):int()
dataset.answers = (torch.rand(10)*969+1):int()
dataset.nsample = 10
dataset.nvocab = 857
dataset.nanswer = 969

local criterion = nn.ClassNLLCriterion()

local opt = {
    batch_size = 4,
    gupid = 0,
    max_epoch = 1,
    quiet = true,
    blind = true,
}
train(opt, model, criterion, dataset)
print 'OK'
