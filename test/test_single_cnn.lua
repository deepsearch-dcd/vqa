require 'nn'
require 'cunn'
local DAQUAR = require 'datasets/DAQUAR'
local npy4th = require 'npy4th'
local model = require 'models/single_cnn'

local function accuracy(net, testset)
	correct = 0
	for i = 1, testset:size() do
		local x, t = unpack(testset[i])
		local y = net:forward(x)
		local confidences, indices = torch.sort(y, true)
		if t == indices[1] then
			correct = correct + 1
		end
	end
	print(correct, 100*correct/testset:size()..'%')
end

local function switch_feature(dataset, fea_vocab)
	im_fea = torch.Tensor(dataset:size(), fea_vocab:size(2))
	for i = 1, dataset:size() do
		im_fea[i] = fea_vocab[dataset[i][1]]
	end
	dataset['images'] = im_fea
end

local function narrow(dataset, size)
	dataset.images = dataset.images[{{1,size}}]
	dataset.questions = dataset.questions[{{1,size},{}}]
	dataset.answers = dataset.answers[{{1,size}}]
	return dataset
end

local function to_cuda(dataset)
	dataset.images = dataset.images:cuda()
	dataset.questions = dataset.questions:cuda()
	dataset.answers = dataset.answers:cuda()
	return dataset
end

-- load dataset
local trainset, testset, vocab = DAQUAR.process_and_check()
local features = npy4th.loadnpy('features/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')

-- switch image index to feature
switch_feature(trainset, features)
switch_feature(testset, features)

model = model:cuda()
local criterion = nn.ClassNLLCriterion():cuda()
to_cuda(trainset)
to_cuda(testset)

for iter = 1, 100 do
	errors = 0
	for i = 1, trainset:size() do
		input, output = unpack(trainset[i])
		errors = errors + criterion:forward(model:forward(input), output)
		model:zeroGradParameters()
		model:backward(input, criterion:backward(model.output, output))
		model:updateParameters(0.001)
	end
	errors = errors / trainset:size()
	print('#current error = '..errors)
	accuracy(model, testset)
end
