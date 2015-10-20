local util = {}

local function _flatten(flat_tt, tt)
        for _, item in pairs(tt) do
                if type(item) ~= 'table' then
                        table.insert(flat_tt, item)
                else
                        _flatten(flat_tt, item)
                end
        end
        return flat_tt
end

-- Flatten the given table.
function util.flatten(tt)
        return _flatten({}, tt)
end

-- Add [items] into [item_to_index], [count] is the last index of [item_to_index]
local function _extract_vocab(items, item_to_index, count)
        for _, item in ipairs(items) do
                if not item_to_index[item] then
                        count = count + 1
                        item_to_index[item] = count
                end
        end
        return item_to_index, count
end

-- extract vocabulay from [items] and add the extra name from [...] to the head of vocabulay.
function util.extract_vocab(items, ...)
        local item_to_index, index_to_item = {}, {}
        local count = 0
        if #{...} > 0 then
                item_to_index, count = _extract_vocab({...},
                                                item_to_index, count)
        end
        item_to_index = _extract_vocab(items, item_to_index, count)
        for item, index in pairs(item_to_index) do
                index_to_item[index] = item
        end
        return item_to_index, index_to_item
end

-- Use the pattern `pt` to split the string `str`
-- pt: [str] define which part of `str` will be reserve.
-- skip: [boolean] if true, will skip the blank part. *default* is true.
local function split(str, pt, skip)
        skip = skip or true
        local ret = {}
        if skip then
                for chunk in string.gfind(str, pt) do
                        table.insert(ret, chunk)
                end
        else
                for chunk in string.gfind(str,pt) do
                        if chunk ~= '' then
                                table.insert(ret, chunk)
                        end
                end
        end
        return ret
end

function util.split_line(str, skip)
        return split(str, '(.-)\n', skip or true)
end

function util.split_word(str)
        return split(str, '%S+')
end

function util.start_with(str, head)
	escapes = {'%(', '%)', '%.', '%+', '%-', '%*', '%?', '%[', '%^', '%$'}
	head = string.gsub(head, '%%', '%%%%')
	for _, esc in ipairs(escapes) do
		head = string.gsub(head, esc, '%%'..esc)
	end
	if string.find(str, '^'..head) then return true
	else return false end
end

-- Compute accuracy of the [net] across the [dataset]
function util.accuracy(net, dataset)
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
function util.to_cuda(dataset)
	for k, v in pairs(dataset) do
		if type(v) == 'userdata' then
			pcall(function() dataset[k] = dataset[k]:cuda() end)
		end
	end
	return dataset
end

-- Narrow [dataset] for test usage
function util.narrow(dataset, size)
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
function util.lookup(indices, table)
	assert(indices:nDimension() == 1)
	ret_size = table:size()
	ret_size[1] = indices:size(1)
	ret = torch.Tensor(ret_size)
	for i = 1, indices:size(1) do
		ret[i] = table[indices[i]]
	end
	return ret
end

return util
