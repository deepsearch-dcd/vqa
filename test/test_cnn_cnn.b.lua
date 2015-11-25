require 'nn'
require 'cunn'
require 'util/train'

local util = require 'util/util'
local model = require 'model/cnn_cnn.b'

print 'test cnn_cnn.b...'
local dataset = {}
dataset.questions = torch.rand(10, 30, 50)
dataset.images = torch.rand(10, 1000)
dataset.answers = (torch.rand(10)*969):int()
dataset.nsample = 10
dataset.nimage = 1669
dataset.nvocab = 857
dataset.nanswer = 969

local criterion = nn.ClassNLLCriterion()

local opt = {
    batch_size = 4,
    max_epoch = 1,
    gpuid = 0,
    quiet = true,
}
train(opt, model, criterion, dataset)

print 'OK'
