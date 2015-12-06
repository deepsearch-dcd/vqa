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
        skip = (skip==nil) or skip
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
        return split(str, '(.-)\n', (skip==nil) or skip)
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

-- Compute accuracy and loss of the [net] across the [dataset]
function util.eval(net, criterion, dataset)
	local correct = 0
	local loss = 0
	for i = 1, dataset:size() do
		local x, t = unpack(dataset[i])
		local y = net:forward(x)
		loss = loss + criterion:forward(y, t)
		local _, indices = torch.max(y, 1)
		if t == indices[1] then
			correct = correct + 1
		end
	end
	return correct/dataset:size(), loss/dataset:size()
end

-- Convert [dataset] into cuda mode
function util.to_cuda(dataset)
	for k, v in pairs(dataset) do
		if type(v) == 'userdata' then
			pcall(function() dataset[k] = dataset[k]:float():cuda() end)
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

-- index dataset
function util.index_dataset(dataset)
    setmetatable(dataset, {__index =
        function(t, i)
            if type(i) == 'number' then
                return {{t.images[{{i}}], t.questions[{{i}}]}, t.answers[{{i}}]}
            end
        end}
    )
   function dataset:size() return self.images:size(1) end

   return dataset
end

-- assemble dataset by replacing id with feature vector(e.g. word embedding)
function util.assemble(ids, vocab)
    assert(#ids:size() > 0)
    if #vocab:size() == 1 then
        vocab = vocab:view(-1,1)
    end
    assert(#vocab:size() == 2)

    -- get creator
    local Tensor = ids.new
    local Storage = ids:size().new

    -- allocate the output tensor
    local out_size = ids:size():totable()
    table.insert(out_size, vocab:size(2))
    local out = Tensor(Storage(out_size))

    -- start assembling
    local flat_ids = ids:view(-1)
    local flat_out = out:view(ids:nElement(), vocab:size(2))
    
    for i = 1, flat_ids:size(1) do
        flat_out[i] = vocab[flat_ids[i]]
    end
    return out
end

return util
