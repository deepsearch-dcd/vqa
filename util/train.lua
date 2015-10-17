local train = {}

-- Compute accuracy of the [net] across the [dataset]
function train.accuracy(net, dataset)
	correct = 0
	for i = 1, dataset:size() do
		local x, t = unpack(dataset[i])
		local y = net:forward(x)
		local _, indices = torch.max(y, 1)
		if t == indices[1] then
			correct = correct + 1
		end
	end
	return correct/dataset:size()
end

-- Convert [dataset] into cuda mode
function train.to_cuda(dataset)
	for k, v in pairs(dataset) do
		if type(v) == 'userdata' then
			pcall(function() dataset[k] = dataset[k]:cuda() end)
		end
	end
	return dataset
end

-- Narrow [dataset] for test usage
function train.narrow(dataset, size)
	assert(dataset:size() >= size)
	for k, v in pairs(dataset) do
		if type(v) == 'userdata' then
			pcall(function() 
				dataset[k] = dataset[k]:narrow(1, 1, size) 
			      end)
		end
	end
	return dataset
end

-- Lookup table, switch indices to the content associated with it
function train.lookup(indices, table)
	assert(indices:nDimension() == 1)
	ret_size = table:size()
	ret_size[1] = indices:size(1)
	ret = torch.Tensor(ret_size)
	for i = 1, indices:size(1) do
		ret[i] = table[indices[i]]
	end
	return ret
end


return train
