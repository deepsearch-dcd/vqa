require 'nn'
require 'cunn'
require 'util/train'

local DAQUAR = require 'dataset/DAQUAR'
--local npy4th = require 'npy4th'
--local model = require 'model/cnn_cnn.a'
local util = require 'util/util'

-- load dataset
local trainset, testset, vocab = DAQUAR.process_and_check()
local features = npy4th.loadnpy('feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')

-- assembly word embedding into dataset
do
	local CBOW = torch.load('word_embedding/CBOW_50.t7')
	local word_to_index, index_to_emb = 
			CBOW.word_to_index, CBOW.index_to_emb
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
            dataset.questions = util.assemble(dataset.questions, torch.Tensor(vocab_to_emb)):resizeAs(dataset.questions)
            dataset.questions = util.assemble(dataset.questions, index_to_emb)
		end
	to_embedding(trainset)
	to_embedding(testset)
end

-- assembly image feature into dataset
trainset.images = util.assemble(trainset.images, features)
testset.images = util.assemble(testset.images, features)

-- train and test
local criterion = nn.ClassNLLCriterion()

local opt = {
    batch_size = 32,
    display_interval = 50,
    gpuid = 0,
    plot_dir = 'done/cnn_cnn.a',
    learningRate = 0.001
}
train(opt, model, criterion, trainset, testset)
