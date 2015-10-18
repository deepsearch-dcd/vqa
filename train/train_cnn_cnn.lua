require 'nn'
require 'cunn'
require 'util/SGDTrainer'

local DAQUAR = require 'dataset/DAQUAR'
local npy4th = require 'npy4th'
local model = require 'model/cnn_cnn'
local util = require 'util/util'

-- load dataset
local trainset, testset, vocab = DAQUAR.process_and_check()
local features = npy4th.loadnpy('feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')

-- switch image index to feature
trainset.images = util.lookup(trainset.images, features)
testset.images = util.lookup(testset.images, features)

model = model:cuda()
local criterion = nn.ClassNLLCriterion():cuda()
util.to_cuda(trainset)
util.to_cuda(testset)

local trainer = SGDTrainer(model, criterion)
trainer:train(trainset, testset)
