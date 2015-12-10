require 'nn'
require 'cunn'
require 'util/train'

local DAQUAR = require 'dataset/DAQUAR'
local npy4th = require 'npy4th'
local model = require 'model/cnn_cnn'
local util = require 'util/util'

-- load dataset
local trainset, testset, vocab = DAQUAR.process_and_check()
local features = npy4th.loadnpy('feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')

-- switch image index to feature
trainset.images = util.assemble(trainset.images, features)
testset.images = util.assemble(testset.images, features)

local criterion = nn.ClassNLLCriterion()

local opt = {
    batch_size = 32,
    display_interval = 500,
    gpuid = 0,
    plot_dir = 'done/cnn_cnn',
    tag = 'apple',
    log_dir = 'done/cnn_cnn',
}
train(opt, model, criterion, trainset, testset)
