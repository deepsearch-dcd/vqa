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
            if chunk ~= '' then
                table.insert(ret, chunk)
            end
        end
    else
        for chunk in string.gfind(str,pt) do
            table.insert(ret, chunk)
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

function util.split_word_with_punc(str)
    return split(str, '[^%s,?";!]+')
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

function util.read_lines(fname)
    local f = assert(io.open(fname, 'r'))
    local file_content = f:read('*all')
    f:close()
    return util.split_line(file_content)
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
		if string.find(string.lower(torch.type(v)), 'tensor') then
			--pcall(function() dataset[k] = dataset[k]:float():cuda() end)
            dataset[k] = dataset[k]:float():cuda()
        elseif torch.type(v) == 'table' then
            util.to_cuda(dataset[k])
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

    -- get creator
    local Tensor = ids.new
    local Storage = ids:size().new

    -- allocate the output tensor
    local out_size = ids:size():totable()
    local flat_out_size = {ids:nElement()}
    for i, dim in ipairs(vocab:size():totable()) do
        if i > 1 then
            table.insert(out_size, dim)
            table.insert(flat_out_size, dim)
        end
    end
    local out = Tensor(Storage(out_size))

    -- start assembling
    local flat_ids = ids:view(-1)
    local flat_out = out:view(Storage(flat_out_size))
    
    for i = 1, flat_ids:size(1) do
        flat_out[i] = vocab[flat_ids[i]]
    end
    return out
end

-- the another version of assemble deal with table
function util.index_data(data, vocab, unk_data)
    if #data == 0 then return end
    for i,sub_data in ipairs(data) do
        if type(sub_data) == 'table' then
            util.index_data(sub_data, vocab, unk_data)
        else
            data[i] = vocab[sub_data]
            if data[i] == nil then
                assert(vocab[unk_data] ~= nil)
                data[i] = vocab[unk_data]
            end
        end
    end
end

function util.load_emb(emb_path)
    local embs = torch.load(emb_path)
    local word_to_emb = embs.word_to_index
    for w, i in pairs(word_to_emb) do
        word_to_emb[w] = embs.index_to_emb[i]
    end
    return word_to_emb
end

local function _to_emb(word_to_emb, words)
    for k, v in pairs(words) do
        if type(v) ~= 'table' then
            assert(type(v) == 'string')
            words[k] = word_to_emb[v]
            if not v then
                words[k] = word_to_emb['*unk*']
            end
        else
            _to_emb(word_to_emb, v)
        end
    end
end

-- replace the word in the table `words` with its embedding, 
-- `words` can have sub table. If a word is unknown use the embedding of *unk*.
function util.to_emb(emb_path, words)
   local word_to_emb = util.load_emb(emb_path) 
   assert(word_to_emb['*unk*'])
   assert(type(words) == 'table')
   _to_emb(word_to_emb, words)
   return words
end

return util
