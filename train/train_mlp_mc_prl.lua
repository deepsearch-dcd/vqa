require 'nn'
npy4th = require 'npy4th'

require 'util/train_mc_prl'

DATASET_PATH = 'dataset/data/VQA/'
TRAINSET_PATH = DATASET_PATH .. 'MC_group_qtype_train2014_25.npy'
VALSET_PATH = DATASET_PATH .. 'MC_group_qtype_val2014_25.npy'

-- load dataset
function load_dataset(path_name)
    local raw = npy4th.loadnpy(path_name)
    local x = raw
    local t = torch.ones(raw:size(1)) * -1
    local dataset = {x = x, t = t}
    function dataset:size() return self.x:size(1) end
    return dataset
end
print('load trainset at ' .. TRAINSET_PATH)
trainset = load_dataset(TRAINSET_PATH)
print('load valset at ' .. VALSET_PATH)
valset = load_dataset(VALSET_PATH)

-- load model and criterion
model = dofile 'model/mlp_mc_prl.lua'
criterion = nn.HingeEmbeddingCriterion(1)

opt = {
    log_dir = 'done/mlp_mc_prl',
    display_interval = 500,
    gpuid = 0,
    plot_dir = 'done/mlp_mc_prl',
    tag = 'default',
    check_point = 1,
    cp_dir = 'done/mlp_mc_prl',
}
train(opt, model, criterion, trainset, valset)
