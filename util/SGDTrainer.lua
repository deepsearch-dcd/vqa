local util = require 'util/util'
local Plot = require 'itorch.Plot'
require 'paths'

local SGDTrainer = torch.class('SGDTrainer')

function SGDTrainer:__init(net, criterion)
	self.net = assert(net)
	self.criterion = assert(criterion)
	self.lr = 0.001         -- learning rate
	self.maxEpoch = 100
	self.displayIter = 500
	self.snapshotIter = 5000
	self.snapshotPrefix = 'iter_'
	self.testIter = 3000
	self.quiet = false    -- don't display any thing
	self.visualPath = '.'
end

local function visual_to_html(train, test, xlabel, save_path)
	local plot_loss = Plot()
	local plot_acc = Plot()
	if #train ~= 0 then
		train = torch.Tensor(train)
		plot_loss:line(train[{{},1}], train[{{},2}], 'red', 'train')
		plot_acc:line(train[{{},1}], train[{{},3}], 'red', 'train')
	end
	if #test ~= 0 then
		test = torch.Tensor(test)
		plot_loss:line(test[{{},1}], test[{{},2}], 'green', 'test')
		plot_acc:line(test[{{},1}], test[{{},3}], 'green', 'test')
	end
	plot_acc:title('Accuracy'):xaxis(xlabel):legend(true):draw()
	plot_acc:save(paths.concat(save_path, xlabel..'_acc.html'))
	plot_loss:title('Loss'):xaxis(xlabel):legend(true):draw()
	plot_loss:save(paths.concat(save_path, xlabel..'_loss.html'))
end

function SGDTrainer:train(trainset, testset)
	local net = self.net
	local criterion = self.criterion
	local totalIter = 0
	local epoch_train_result, epoch_test_result = {}, {}
	local iter_train_result, iter_test_result = {}, {}
	local iter_loss, iter_correct = 0, 0
	local epoch_loss, epoch_correct = 0, 0
	local max_train_acc, max_test_acc = 0, 0
	for epoch = 1, self.maxEpoch do
		local shuffled_indices = torch.randperm(trainset:size())
		if not self.shuffle then
			for i = 1, trainset:size() do
				shuffled_indices[i] = i
			end
		end
		for iter = 1, trainset:size() do
			x, t = unpack(trainset[shuffled_indices[iter]])
			loss = criterion:forward(net:forward(x), t)
			iter_loss = iter_loss + loss
			epoch_loss = epoch_loss + loss
			local _, indices = torch.max(net.output, 1)
			if t == indices[1] then
				iter_correct = iter_correct + 1
				epoch_correct = epoch_correct + 1
			end
			net:zeroGradParameters()
			net:backward(x, criterion:backward(net.output, t))
			net:updateParameters(self.lr)

			totalIter = totalIter + 1
			
			-- display
			if not self.quiet and self.displayIter
				and totalIter % self.displayIter == 0 then
				iter_loss = iter_loss / self.displayIter
				iter_correct = iter_correct / self.displayIter
				if iter_correct > max_train_acc then
					max_train_acc = iter_correct
				end
				print(string.format('# Train Epoch %d,'
					..' Iteration %d, lr = %.6f', 
					epoch, totalIter, self.lr))
				print(string.format('\tloss = %.2f', iter_loss))
				print(string.format('\tacc = %.2f%%(%.2f%%)', 
					100*iter_correct, 100*max_train_acc))
				table.insert(iter_train_result, {totalIter, 
						iter_loss, iter_correct})
				iter_loss, iter_correct = 0, 0
			end
			-- test
			if not self.quiet and testset and self.testIter
				and totalIter % self.testIter == 0 then
				local test_acc, test_loss = util.eval(net, 
							criterion, testset)
				if test_acc > max_test_acc then
					max_test_acc = test_acc
				end
				print(string.format('# Test Epoch %d,'
					..' Iteration %d', epoch, totalIter))
				print(string.format('\tloss = %.2f', test_loss))
				print(string.format('\tacc = %.2f%%(%.2f%%)', 
					100*test_acc, 100*max_test_acc))
				table.insert(iter_test_result, {totalIter,
					test_loss, test_acc})
			end
			-- visualize
			if not self.quiet and self.visualPath then
				if #iter_train_result ~= 0 or
					#iter_test_result ~= 0 then
					visual_to_html(iter_train_result,
						iter_test_result, 'Iteration', 
						self.visualPath)
				end
			end
			-- snapshot
			if self.snapshotIter and 
				totalIter % self.snapshotIter == 0 then
				torch.save(self.snapshotPrefix..totalIter
						..'.t7', net)
			end
			--
		end
		epoch_loss = epoch_loss / trainset:size()
		epoch_correct = epoch_correct / trainset:size()
		if not self.quiet then
			print(string.format('# Epoch %d Training', epoch))
			print(string.format('\tloss = %.2f', epoch_loss))
			print(string.format('\tacc = %.2f%%', 
						100*epoch_correct))
			table.insert(epoch_train_result, {epoch, epoch_loss, 
					epoch_correct})
			epoch_loss, epoch_correct = 0, 0
			if testset then
				local test_acc, test_loss = util.eval(net, 
							criterion, testset)
				print(string.format('# Epoch %d Testing', 
							epoch))
				print(string.format('\tloss=%.2f', test_loss))
				print(string.format('\tacc = %.2f%%', 
							test_acc*100))
				table.insert(epoch_test_result, {epoch,
						test_loss, test_acc})
			end
			-- visual
			if self.visualPath then
				if #epoch_train_result ~= 0 or
					#epoch_test_result ~= 0 then
					visual_to_html(epoch_train_result,
						epoch_test_result, 'Epoch',
						self.visualPath)
				end
			end
		end
	end
end
