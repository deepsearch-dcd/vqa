require 'paths'
npy4th = require 'npy4th'

require 'util/train'
util = require 'util/util'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training tool for VQA model')
cmd:text()
cmd:text('Options')
cmd:option('-model', 'default', 'the file of VQA model')
cmd:option('-settings', 'default', 'the file contains settings')
cmd:option('-log_dir', 'default', 'where the log save')
cmd:text()

params = cmd:parse(arg)

-- load settings to globle table
-- dataset: DAQUAR | COCOQA
-- feature_path: [optional] path of *.npy
-- embedding_path: [optional] path of embeddings
-- criterion: criterion will be used to train model
-- opt: used for train()
assert(paths.filep(params.settings))
dofile(params.settings)

-- load dataset
assert(dataset)
if dataset == 'DAQUAR' then
    local DAQUAR = require 'dataset/DAQUAR'
    trainset, testset, vocab = DAQUAR.process_and_check()
end

-- load features
if feature_path ~= nil then
    local features = npy4th.loadnpy(feature_path)
    trainset.images = util.assemble(trainset.images, features)
    testset.images = util.assemble(testset.images, features)
end

-- load embeddings
if embedding_path ~= nil then
    local embeddings = torch.load(embedding_path)
    local word_to_index, index_to_emb =
        embeddings.word_to_index, embeddings.index_to_emb
    local vocab_to_emb = {} -- vocab index to emb index
    local unk_count = 0
    for i, w in ipairs(vocab.index_to_word) do
        local emb_index = word_to_index[w]
        if emb_index then
            vocab_to_emb[i] = emb_index
        else
            unk_count = unk_count + 1
            vocab_to_emb[i] = word_to_index['*unk*']
        end
    end
    print(string.format('unkown word: %d', unk_count))

    local to_embedding =
        function(dataset)
            dataset.questions = util.assemble(dataset.questions, 
                                              torch.Tensor(vocab_to_emb))
                                                :resizeAs(dataset.questions)
            dataset.questions = util.assemble(dataset.questions, index_to_emb)
        end
    to_embedding(trainset)
    to_embedding(testset)
end

-- check criterion
assert(criterion)

-- check opt
assert(opt)

-- set dir for opt
opt.plot_dir = 'done/' .. params.log_dir
opt.log_dir = opt.plot_dir
paths.mkdir(opt.log_dir)

--check model
assert(paths.filep(params.model))
model = dofile(params.model)

train(opt, model, criterion, trainset, testset)
