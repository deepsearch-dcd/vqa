--[[
-- Usage:
-- COCOQA = require 'dataset/COCOQA.lua'
-- trainset, testset, vocab = COCOQA.load_data{
--      format='table', -- or tensor
--      use_pad_word=false,
--      use_unk_word=true,
--      use_unk_answer=false}
--]]
require 'paths'
local torch = require 'torch'

util = require 'util/util'

local COCOQA = {}

-- special word as placeholder
local UNK_WORD, PAD_WORD = '*unk*', '*pad*'

-- dataset dir
local DATASET_DIR = 'dataset/data/COCO-QA'
local TRAIN_DIR = DATASET_DIR .. '/train'
local TEST_DIR = DATASET_DIR .. '/test'

-- dataset file name
local IMAGES_FILE_NAME = 'img_ids.txt'
local QUESTIONS_FILE_NAME = 'questions.txt'
local ANSWERS_FILE_NAME = 'answers.txt'
local TYPES_FILE_NAME = 'types.txt'

-- question type indexing
local TYPE_TO_INDEX = {object=1, number=2, color=3, location=4}
local INDEX_TO_TYPE = {[1]='object', [2]='number', [3]='color', [4]='location'}

-- captions file
local CAPTIONS_FILE = 'dataset/data/COCO-QA/done/captions.txt'

-- determiner
local DETERMINER = {
    ['the']=0, ['a']=0, ['an']=0, ['my']=0, ['your']=0, 
    ['his']=0, ['her']=0, ['our']=0, ['their']=0, ["one's"]=0, 
    ['its']=0, ['this']=0, ['that']=0, ['these']=0, ['those']=0, 
    ['such']=0, ['some']=0, ['any']=0, ['each']=0, ['every']=0, 
    ['enough']=0, ['either']=0, ['neither']=0, ['all']=0, ['both']=0, 
    ['half']=0, ['several']=0, ['many']=0, ['much']=0, ['few']=0, 
    ['little']=0, ['other']=0, ['another']=0}

--[[
-- Read from file and save in table.
--]]
local function process_raw(file_dir)
    
    local function read_file(fname)
        local f = assert(io.open(fname, 'r'))
        local file_content = f:read('*all')
        f:close()
        return util.split_line(file_content)
    end

    -- process images
    local fname = paths.concat(file_dir, IMAGES_FILE_NAME)
    local images = read_file(fname)

    -- process questions
    fname = paths.concat(file_dir, QUESTIONS_FILE_NAME)
    local lines = read_file(fname)
    local questions = {}
    for _,l in ipairs(lines) do
        table.insert(questions, util.split_word(l))
    end
    assert(#images == #questions)

    -- process answers
    fname = paths.concat(file_dir, ANSWERS_FILE_NAME)
    local answers = read_file(fname)
    assert(#questions == #answers) 

    -- process types
    fname = paths.concat(file_dir, TYPES_FILE_NAME)
    local types = read_file(fname)
    assert(#answers == #types)

    return images, questions, answers, types

end


local function process_caption()
        
    local f = assert(io.open(CAPTIONS_FILE, 'r'))
    local lines = util.split_line(f:read('*all'))
    f:close()

    local captions = {}
    for _,l in ipairs(lines) do
        table.insert(captions, util.split_word(l))
    end

    -- normalize words
    --  remove punctuation at the end of last word
    --  capital case to lower case
    for i, cap in ipairs(captions) do
        for j, w in ipairs(cap) do
            cap[j] = string.lower(w)
        end
        cap[#cap] = string.gsub(cap[#cap], '(.-)%p$', '%1')
    end
    return captions

end


local function discard_word(questions, discarded)
    
    local new_questions = {}
    for i,q in ipairs(questions) do
        new_questions[i] = {}
        for _,w in ipairs(q) do
            if discarded[w] == nil then
                table.insert(new_questions[i], w)
            else
                discarded[w] = discarded[w] + 1
            end
        end
    end
    return new_questions

end


local function pad_or_chop(questions, max_length, pad_word)

    for _,q in ipairs(questions) do
        len = #q
        if len <= max_length then
            for i = len+1, max_length do
                q[i] = pad_word
            end
        else
            for i = max_length+1, len do
                q[i] = nil
            end
        end
        assert(#q == max_length)
    end
    
end


local function add_statistic(dataset, vocab)
    
    dataset.nsample = #dataset.questions
    dataset.nimage = #vocab.index_to_image
    dataset.nvocab = #vocab.index_to_word
    dataset.nanswer = #vocab.index_to_answer

end


local function format_tensor(dataset)
    
    local Tensor = torch.Tensor
    dataset.images = Tensor(dataset.images)
    dataset.questions = Tensor(dataset.questions)
    dataset.answers = Tensor(dataset.answers)
    dataset.types = Tensor(dataset.types)

end


function COCOQA.load_data(settings)
    -- :settings:
    -- format: table or tensor
    -- add_pad_word: if true add PAD_WORD to vocab and pad the question to 
    --               max_length
    -- add_unk_word: if true add UNK_WORD to vocab
    -- add_unk_answer: if true treat unseen answer in testset as UNK_WORD
    -- max_length
    -- discard_det: if true discard the words in DETERMINER from questions
    -- load_caption: if true, load caption additionally

    -- load raw data
    local train_images , train_questions, train_answers, train_types = 
        process_raw(TRAIN_DIR)
    local test_images, test_questions, test_answers, test_types = 
        process_raw(TEST_DIR)
    local captions = {}
    if settings.load_caption then
        captions = process_caption()
    end

    if settings.discard_det == true then
        train_questions = discard_word(train_questions, DETERMINER)
        test_questions = discard_word(test_questions, DETERMINER)
        captions = discard_word(captions, DETERMINER)
    end

    -- extract vocabulary
    local vocab = {}
    vocab.image_to_index, vocab.index_to_image = 
        util.extract_vocab(util.flatten({train_images, test_images}))
    vocab.word_to_index, vocab.index_to_word = 
        util.extract_vocab(util.flatten({train_questions, captions}))
    vocab.answer_to_index, vocab.index_to_answer = 
        util.extract_vocab(train_answers)
    vocab['type_to_index'], vocab['index_to_type'] = 
        TYPE_TO_INDEX, INDEX_TO_TYPE
    if settings.add_unk_word then
        table.insert(vocab.index_to_word, UNK_WORD)
        vocab.word_to_index[UNK_WORD] = #vocab.index_to_word
    end
    if settings.add_pad_word then
        table.insert(vocab.index_to_word, PAD_WORD)
        vocab.word_to_index[PAD_WORD] = #vocab.index_to_word
    end
    if settings.add_unk_answer then
        table.insert(vocab.index_to_answer, UNK_WORD)
        vocab.answer_to_index[UNK_WORD] = #vocab.index_to_answer
    end

    -- index data
    util.index_data(train_images, vocab.image_to_index)
    util.index_data(train_questions, vocab.word_to_index)
    util.index_data(train_answers, vocab.answer_to_index)
    util.index_data(test_images, vocab.image_to_index)
    util.index_data(test_questions, vocab.word_to_index, UNK_WORD)
    util.index_data(test_answers, vocab.answer_to_index, UNK_WORD)
    util.index_data(captions, vocab.word_to_index)

    -- padding
    if settings.add_pad_word then
        assert(settings.max_length)
        local padding = vocab.word_to_index[PAD_WORD]
        assert(padding)
        pad_or_chop(train_questions, settings.max_length, padding)
        pad_or_chop(test_questions, settings.max_length, padding)
        pad_or_chop(captions, settings.max_length, padding)
    end

    -- align captions to images
    local cap_ = {}
    for i = 1,#captions,5 do
        table.insert(cap_, {captions[i], captions[i+1], captions[i+2],
                               captions[i+3], captions[i+4]})
    end
    captions = cap_
    if settings.load_caption then
        assert(#captions == #vocab.index_to_image)
    end
    captions = cap_
    if settings.load_caption then
        assert(#captions == #vocab.index_to_image)
    end

    -- constructure dataset
    local trainset = {
        images = train_images,
        questions = train_questions,
        answers = train_answers,
        types = train_types,
        captions = captions}
    local testset = {
        images = test_images, 
        questions = test_questions,
        answers = test_answers,
        types = test_types}

    -- add statistic of the dataset
    add_statistic(trainset, vocab)
    add_statistic(testset, vocab)

    -- formatting
    local Tensor = torch.Tensor
    if settings.format == 'tensor' then
        format_tensor(trainset)
        format_tensor(testset)
    end

    return trainset, testset, vocab

end

return COCOQA
