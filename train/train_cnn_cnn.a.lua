require 'nn'
require 'cunn'
require 'util/SGDTrainer'

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
	local vocab_to_emb = {}
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
			local q_count = dataset.questions:size(1)
			-- a question has 30 word.
			local flat_q = dataset.questions:resize(q_count*30)
			flat_q = util.lookup(flat_q, torch.Tensor(vocab_to_emb))
			flat_q = util.lookup(flat_q, index_to_emb) 
			-- word embedding dimension is 50
			dataset.questions = flat_q:resize(q_count, 30, 50)
		end
	to_embedding(trainset)
	to_embedding(testset)
end

-- assembly image feature into dataset
trainset.images = util.lookup(trainset.images, features)
testset.images = util.lookup(testset.images, features)

-- train and test
model = model:cuda()
local criterion = nn.ClassNLLCriterion():cuda()
util.to_cuda(trainset)
util.to_cuda(testset)

local trainer = SGDTrainer(model, criterion)
trainer:train(trainset, testset)
