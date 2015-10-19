local util = require 'util/util'

local SGDTrainer = torch.class('SGDTrainer')

function SGDTrainer:__init(net, criterion)
	self.net = assert(net)
	self.criterion = assert(criterion)
	self.lr = 0.001
	self.maxEpoch = 100
	self.displayIter = 500
	self.snapshotIter = 5000
	self.snapshotPrefix = 'iter_'
	self.verbose = true
end

function SGDTrainer:train(trainset, testset)
	local net = self.net
	local criterion = self.criterion
	local totalIter = 0
	for epoch = 1, self.maxEpoch do
		local loss = 0
		local correct = 0
		for iter = 1, trainset:size() do
			x, t = unpack(trainset[iter])
			loss = loss + criterion:forward(net:forward(x), t)
			local _, indices = torch.max(net.output, 1)
			if t == indices[1] then
				correct = correct + 1
			end
			net:zeroGradParameters()
			net:backward(x, criterion:backward(net.output, t))
			net:updateParameters(self.lr)

			totalIter = totalIter + 1
			-- display
			if self.verbose and self.displayIter and totalIter % self.displayIter == 0 then
				print(string.format('# Epoch %d, Iteration %d, lr = %.6f', epoch, totalIter, self.lr))
				print(string.format('\tloss = %.2f', loss/iter))
				print(string.format('\tacc = %.2f%%', 100*correct/iter))
			end
			-- snapshot
			if self.snapshotIter and totalIter % self.snapshotIter == 0 then
				torch.save(self.snapshotPrefix..totalIter..'.t7', net)
			end
		end
		if self.verbose then
			print(string.format('# Epoch %d Training', epoch))
			print(string.format('\tloss = %.2f', 
						loss/trainset:size()))
			print(string.format('\tacc = %.2f%%', 
						100*correct/trainset:size()))
		end
		if self.verbose and testset then
			local acc = util.accuracy(net, testset)
			print(string.format('# Epoch %d Testing', epoch))
			print(string.format('\tacc = %.2f%%', acc*100))
		end
	end
end	
