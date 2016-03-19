require 'nn'
require 'cunn'

local npy4th = require 'npy4th'
local util = require 'util/util'

require 'util/train_gen'

-- dataset
print('load dataset..')
local COCOQA = require 'dataset/COCOQA'
local trainset, testset, vocab = COCOQA.load_data{
    format='table',
    add_unk_word=true,
    add_pad_word=false,
    top_word=999,    -- use for ba_cnn_cnn.a
    add_unk_answer=false}
--]]
-- create dummy data
--[[
local function dummy_dataset()
    local dataset = {nsample=10, nimage=10, nvocab=10, nanswer=430}
    dataset.images = torch.totable(
                        (torch.rand(dataset.nsample)*dataset.nimage + 1):int())
    dataset.questions = {}
    for i = 1,dataset.nsample do
        table.insert(dataset.questions, 
            torch.totable(
                (torch.rand(torch.random(1,54))*dataset.nvocab+1):int()))
    end
    dataset.answers = torch.totable(
                        (torch.rand(dataset.nsample)*dataset.nanswer+1):int())
    return dataset
end
local trainset = dummy_dataset()
local testset = dummy_dataset()
--]]

-- assemble image features
print('load image features..')
local im_feas = npy4th.loadnpy('feature/COCO-QA/VGG19-512x14x14.npy')
--local im_feas = torch.rand(10, 512, 14, 14)

local wrap = require 'util/COCODatasetWrapper'
wrap(trainset, im_feas, nil, false)
wrap(testset, im_feas, nil, false)

-- model
print('create model..')
local MODEL_NAME = 'ba_cnn_cnn'
local MODEL_FILE = 'model/' .. MODEL_NAME .. '.lua'
local criterion = nn.ClassNLLCriterion()
local model = dofile (MODEL_FILE)
local model_file = io.open(MODEL_FILE, 'r'):read('*all')

-- train
print('training..')
opt = {
    seed = 1234,
    gpuid = 0,
    home_dir = 'done/' .. MODEL_NAME,
    log_dir = '',
    plot_dir = '',
    cp_dir = '',
    tag = 'default',
    display_interval = 10000,
    quiet = false,
    --max_epoch = 1,
    batch_size = 1,
    learningRate = 0.1,
    weightDecay = 0.0005,
    momentum = 0.9,
    --check_point = 1,
    --pretrained_model = '',
    model_def = '\n' .. model_file,
}
train(opt, model, criterion, trainset, testset)
