require 'nn'
require 'cunn'
require 'util/train'

local util = require 'util/util'
local model = require 'model/cnn_cnn'

print 'test cnn_cnn...'
local dataset = {}
dataset.questions = (torch.rand(10, 30)*857):int()
dataset.images = torch.rand(10, 1000)
dataset.answers = (torch.rand(10)*969):int()
dataset.nsample = 10
dataset.nvocab = 857
dataset.nimage = 1669
dataset.nanswer = 969

local criterion = nn.ClassNLLCriterion()

local opt = {
    batch_size = 4,
    gpuid = 0,
    max_epoch = 1,
    quiet = true,
}
train(opt, model, criterion, dataset)
print 'OK'
