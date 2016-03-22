require 'nn'
require 'cunn'

local npy4th = require 'npy4th'
local util = require 'util/util'

require 'util/train_gen'

local DEBUG = false

-- dataset
print('load dataset..')
local trainset, testset, vocab
if not DEBUG then
    local COCOQA = require 'dataset/COCOQA'
    trainset, testset, vocab = COCOQA.load_data{
        format='table',
        add_unk_word=true,
        add_pad_word=false,
        top_word=999,    -- use for ba_cnn_cnn.a
        add_unk_answer=false}
else
    -- create dummy data
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
    trainset = dummy_dataset()
    testset = dummy_dataset()
end

-- assemble image features
print('load image features..')
local im_feas
if not DEBUG then
    im_feas = npy4th.loadnpy('feature/COCO-QA/VGG19-1000.npy')
    --im_feas = npy4th.loadnpy('feature/COCO-QA/VGG19-512x14x14.npy')
else
    im_feas = torch.rand(10, 1000)
    --im_feas = torch.rand(10, 512, 14, 14)
end

local wrap = require 'util/COCODatasetWrapper'
wrap(trainset, im_feas, 'blind', true)
wrap(testset, im_feas, 'blind', true)

-- model
print('create model..')
local MODEL_NAME = 'COCO_emb_null'
local MODEL_FILE = 'model/' .. MODEL_NAME .. '.lua'
local criterion = nn.ClassNLLCriterion()
local model = dofile (MODEL_FILE)
local model_file = io.open(MODEL_FILE, 'r'):read('*all')

-- train
print('training..')
opt = {
    seed = 1234,
    gpuid = 0,
    test=DEBUG,
    home_dir = 'done/' .. MODEL_NAME,
    log_dir = '',
    plot_dir = '',
    cp_dir = '',
    tag = 'default',
    display_interval = 10000,
    quiet = false,
    --max_epoch = 1,
    batch_size = 1,
    learningRate = 0.001,
    weightDecay = 0.0005,
    momentum = 0.9,
    --check_point = 1,
    --pretrained_model = '',
    model_def = '\n' .. model_file,
}
train(opt, model, criterion, trainset, testset)
