require 'paths'
local torch = require 'torch'

local util = require 'util/util'


-- special word as placeholder
local UNK_WORD = '*unk*'
-- dataset dir
local dataDir = 'dataset/data/VQA/done'

local SubVQA = {}


--[[
-- Read raw data from file.
--]]
local function readRaw(spltName, suffix)
    -- images
    local imgFile = paths.concat(dataDir, spltName, 'image-'..suffix)
    local imgs = util.read_lines(imgFile)
    
    -- questions
    local quesFile = paths.concat(dataDir, spltName, 'question-'..suffix)
    local ques = util.read_lines(quesFile)
    for i,q in ipairs(ques) do
        ques[i] = util.split_word(q)
    end
    assert(#imgs == #ques)

    -- answers
    local ansFile = paths.concat(dataDir, spltName, 'answer-'..suffix)
    local ans = util.read_lines(ansFile)
    assert(#ques == #ans)

    -- types
    local typeFile = paths.concat(dataDir, spltName, 'type-'..suffix)
    local types = util.read_lines(typeFile)
    for i,t in ipairs(types) do
        types[i] = tonumber(t)+1
    end
    assert(#ans == #types)
    return imgs, ques, ans, types
end


local function addInfo(dataset, vocab)
    dataset.nsample = #dataset.questions
    dataset.size = dataset.nsample
    dataset.nimage = #vocab.idx2Img
    dataset.nvocab = #vocab.idx2Word
    dataset.nanswer = #vocab.idx2Ans
end


function SubVQA.load_data(opts)
    -- `settings`
    -- name: optional, the name of the subdataset. e.g. the name of 'question-1000-500-43-5.txt' is '1000-500-43-5'.
    --       default is '2000-1000-27-10'.
    -- vSize: optional, restrict the size of vocabulary as `vSize` with most frequent `vSize` words.
    --        Other words are treated as UNK_WORD.

    
    -- initialize arguments
    opts.name = opts.name or '2000-1000-27-10'

    -- load raw data
    local suffix = opts.name .. '.txt'
    local trnImgs, trnQues, trnAns, trnTypes = readRaw('train', suffix)
    local tstImgs, tstQues, tstAns, tstTypes = readRaw('test', suffix)

    -- extract vocabularies of images, words in questions, answers and question types.
    local vocab = {}
    vocab.img2Idx, vocab.idx2Img = util.extract_vocab(util.flatten(
            {trnImgs, tstImgs}))
    vocab.word2Idx, vocab.idx2Word = util.extract_vocab(util.flatten(trnQues))
    vocab.ans2Idx, vocab.idx2Ans = util.extract_vocab(trnAns)
    cateFile = paths.concat(dataDir, 'cate', 'cate-'..suffix)
    vocab.idx2Type = util.read_lines(cateFile)
    vocab.type2Idx = {}
    for i,t in ipairs(vocab.idx2Type) do
        vocab.type2Idx[t] = i
    end

    -- resize word vocabulary
    if opts.vSize then
        local wordAndCnt = {}
        for i,w in ipairs(vocab.idx2Word) do
            wordAndCnt[w] = 0
        end
        for i,q in ipairs(trnQues) do
            for j,w in ipairs(q) do
                wordAndCnt[w] = wordAndCnt[w] + 1
            end
        end
        local wordByCnt = {}
        for w,c in pairs(wordAndCnt) do
            table.insert(wordByCnt, {w, c})
        end
        table.sort(wordByCnt, function(a, b) return a[2]>b[2] end)
        local idx2Word, word2Idx = {}, {}
        for i=1,opts.vSize do
            w = wordByCnt[i][1]
            idx2Word[i] = w
            word2Idx[w] = i
        end
        vocab.idx2Word, vocab.word2Idx = idx2Word, word2Idx
    end
    table.insert(vocab.idx2Word, UNK_WORD)
    vocab.word2Idx[UNK_WORD] = #vocab.idx2Word

    -- replace the real data with the corresponding index in vocabulary
    util.index_data(trnImgs, vocab.img2Idx)
    util.index_data(trnQues, vocab.word2Idx)
    util.index_data(trnAns, vocab.ans2Idx)
    util.index_data(tstImgs, vocab.img2Idx)
    util.index_data(tstQues, vocab.word2Idx, UNK_WORD)
    util.index_data(tstAns, vocab.ans2Idx)

    -- construct dataset
    local trainset = {images = trnImgs, questions = trnQues, 
                      answers = trnAns, types = trnTypes}
    local testset = {images = tstImgs, questions = tstQues,
                     answers = tstAns, types = tstTypes}
    addInfo(trainset, vocab)
    addInfo(testset, vocab)

    return trainset, testset, vocab
end

return SubVQA
