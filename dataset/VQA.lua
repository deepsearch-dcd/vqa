require 'paths'
local torch = require 'torch'
require 'cunn'

local util = require 'util/util'


-- special word as placeholder
local UNK_WORD = '*unk*'
local START_WORD = '*start*'
local END_WORD = '*end*'
-- dataset dir
local dataDir = 'dataset/data/VQA/done'

local VQA = {}


--[[
-- Read raw data from file.
--]]
local function readRaw(spltName)
    local function read(fName)
        fPath = paths.concat(dataDir, spltName, fName)
        return util.read_lines(fPath)
    end
    local function map(func, atable)
        for k, v in pairs(atable) do
            atable[k] = func(v)
        end
        return atable
    end

    -- images
    local imgs = map(tonumber, read('images.txt'))
    local ques = map(util.split_word, read('questions.txt'))
    local ans = map(util.split_word, read('answers.txt'))
    local qtypes = read('qtypes.txt')
    local atypes = read('atypes.txt')

    assert(#imgs == #ques)
    assert(#ques == #ans)
    assert(#ans == #qtypes)
    assert(#qtypes == #atypes)
    
    return imgs, ques, ans, qtypes, atypes
end

--[[
-- Add START_WORD and END_WORD to the two ends of each answer.
--]]
local function pack(ans)
    for i,_ in ipairs(ans) do
        table.insert(ans[i], 1, START_WORD)
        table.insert(ans[i], END_WORD)
    end
end

local function toTensor(dataset)
    local Tensor = torch.Tensor
    dataset.images = Tensor(dataset.images)
    for i,q in ipairs(dataset.questions) do
        dataset.questions[i] = Tensor(q)
    end
    for i,a in ipairs(dataset.answers) do
        dataset.answers[i] = Tensor(a)
    end
end

function VQA.load(opts)
    -- `settings`
    -- vSize: optional, restrict the size of vocabulary as `vSize` with most fequent `vSize` words.
    --        Other words are treated as UNK_WORD.
    

    -- put extra information into `vocab`
    local vocab = {}

    -- initialize arguments
    local trnImgs, trnQues, trnAns, trnQtypes, trnAtypes = readRaw('train')
    local tstImgs, tstQues, tstAns, tstQtypes, tstAtypes = readRaw('test')

    -- add start word and end word
    vocab.START_WORD = START_WORD
    vocab.END_WORD = END_WORD
    pack(trnAns)
    pack(tstAns)
    vocab.trnAns = trnAns
    vocab.tstAns = tstAns

    -- extract vocabluaries
    vocab.img2Idx, vocab.idx2Img = util.extract_vocab(util.flatten(
            {trnImgs, tstImgs}))
    vocab.word2Idx, vocab.idx2Word = util.extract_vocab(util.flatten(
            {trnQues, trnAns}))

    -- resize word vocabulary
    if opts.vSize then
        local wordAndCnt = {}
        for _,w in ipairs(vocab.idx2Word) do
            wordAndCnt[w] = 0
        end
        for _,q in ipairs(trnQues) do
            for _,w in ipairs(q) do
                wordAndCnt[w] = wordAndCnt[w] + 1
            end
        end
        for _,a in ipairs(trnAns) do
            for _,w in ipairs(a) do
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
    vocab.UNK_WORD = UNK_WORD

    -- replace the real data with the corresponding index in vocabulary
    util.index_data(trnImgs, vocab.img2Idx)
    util.index_data(trnQues, vocab.word2Idx, UNK_WORD)
    util.index_data(trnAns, vocab.word2Idx, UNK_WORD)
    util.index_data(tstImgs, vocab.img2Idx)
    util.index_data(tstQues, vocab.word2Idx, UNK_WORD)
    util.index_data(tstAns, vocab.word2Idx, UNK_WORD)

    -- construct dataset
    local trainset = {images = trnImgs, questions = trnQues, answers = trnAns}
    local testset = {images = tstImgs, questions = tstQues, answers = tstAns}
    toTensor(trainset)
    toTensor(testset)

    return trainset, testset, vocab
end

function VQA.wrap(dataset)
    function dataset:size()
        return #self.questions
    end

    dataset._idx = 0

    function dataset:reset()
        self._idx = 0
    end

    function dataset:next()
        self._idx = self._idx + 1
        if self._idx > self:size() then
            return nil
        end
        local idx = self._idx
        if cuda then
            return self.images[idx], self.questions[idx]:cuda(), 
                   self.answers[idx]:cuda()
        else 
            return self.images[idx], self.questions[idx], self.answers[idx]
        end
    end

    function dataset:cuda()
        self.cuda = true
    end
end

function VQA.textForm(idxes, vocab)
    local START = vocab.START_WORD
    local END = vocab.END_WORD
    local idx2Word = vocab.idx2Word
    local ret = {}
    for i,item in ipairs(idxes) do
        local words = {}
        for j,idx in ipairs(item) do
            words[j] = idx2Word[idx]
        end
        ret[i] = words
    end
    for i,item in ipairs(ret) do
        local from, to = 1, #item
        if item[from] == START then
            from = from + 1
        end
        if item[to] == END then
            to = to - 1
        end
        ret[i] = table.concat(ret[i], ' ', from, to)
    end
    return ret
end

return VQA
