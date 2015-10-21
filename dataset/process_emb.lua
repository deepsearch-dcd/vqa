local npy4th = require 'npy4th'
local util = require 'util/util'

local function convert(words_src, embs_src, dst)
	local f = io.open(words_src, 'r')
	local words = util.split_line(f:read('*all'))
	f:close()
	
	local word_to_index = {}
	for i, w in ipairs(words) do
		word_to_index[w] = i
	end
	print(embs_src)	
	local embs = npy4th.loadnpy(embs_src)
	torch.save(dst, {
		word_to_index = word_to_index,
		index_to_emb = embs
	})
end

local function process()
	local src_dir = 'word_embedding'
	local fnames = {'CBOW_50', 'CBOW_100', 'CBOW_200', 'CBOW_300',
		'CBOW_500', 'SG_50', 'SG_100', 'SG_200', 'SG_300', 'SG_500',
		'HLBL_50'}

	for _, fn in ipairs(fnames) do
		local words_src = string.format('%s/%s_words.txt', src_dir, fn)
		local embs_src = string.format('%s/%s_embs.npy', src_dir, fn)
		local dst = string.format('%s/%s.t7', src_dir, fn)
		convert(words_src, embs_src, dst)
	end
end

do
	process()
end
