--[[
-- Usage:
-- COCOQA = require 'dataset/COCOQA.lua'
-- trainset, testset, vocab = COCOQA.load_data{
--      format='table', -- or tensor
--      add_pad_word=false,
--      add_unk_word=true,
--      add_unk_answer=false}
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
local ORIGIN_CAPTIONS_FILE = 'dataset/data/COCO-QA/done/captions.txt'
local GENERATED_CAPTIONS_FILE = 
    'dataset/data/COCO-QA/done/captions_neuraltalk2_samplemax.txt'

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
local function process_raw(file_dir, filter_type)
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
    for i, _type in ipairs(types) do
        types[i] = tonumber(_type)+1
    end
    assert(#answers == #types)

    local all_images = images
    local filter_index = {}

    local n_images, n_questions, n_answers, n_types = {}, {}, {}, {}
    if filter_type ~= nil then
        for i, _type in ipairs(types) do
            if _type == filter_type then
                table.insert(filter_index, i)
                table.insert(n_images, images[i])
                table.insert(n_questions, questions[i])
                table.insert(n_answers, answers[i])
                table.insert(n_types, types[i])
            end
        end
    else
        n_images = images
        n_questions = questions
        n_answers = answers
        n_types = types
    end

    return n_images, n_questions, n_answers, n_types, all_images, filter_index
end


local function process_caption(source)
    local f = assert(io.open(source, 'r'))
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


local function dup_cap(caps, img_ids)
    local captions = {}
    for _,img_ids in ipairs(img_ids) do
        table.insert(captions, caps[img_ids])
    end
    return captions
end


local function add_statistic(dataset, vocab)
    dataset.nsample = #dataset.questions
    dataset.size = dataset.nsample -- lin
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
    if dataset.captions ~= nil then
        dataset.captions = Tensor(dataset.captions)
    end

    if dataset.tfidfs then
        dataset.tfidfs = Tensor(dataset.tfidfs)
    end
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
    -- load_caption: nil, do nothing; 'origin', load caption from ms coco;
    --               'generate', load caption from generated caption source.
    -- top_word: if not nil, collect the top [top_word] words as the word 
    --              vocabulary.
    -- tfidf: if given, compute tf-idf for each word in question.
    -- bow: if given, the question is represented as BoW form.
    -- filter_type: [object, number, color, location] return questions of specific type.
    -- discard_unk_answer: if true, discard unknown answer in testset.

    -- load raw data
    local train_images , train_questions, train_answers, 
          train_types, train_all_images, train_filter_index = 
            process_raw(TRAIN_DIR, TYPE_TO_INDEX[settings.filter_type])
    local test_images, test_questions, test_answers, 
          test_types, test_all_images, test_filter_index = 
            process_raw(TEST_DIR, TYPE_TO_INDEX[settings.filter_type])
    local captions = {}
    if settings.load_caption == 'origin' then
        captions = process_caption(ORIGIN_CAPTIONS_FILE)
    elseif settings.load_caption == 'generate' then
        captions = process_caption(GENERATED_CAPTIONS_FILE)
    end

    if settings.discard_det == true then
        train_questions = discard_word(train_questions, DETERMINER)
        test_questions = discard_word(test_questions, DETERMINER)
        captions = discard_word(captions, DETERMINER)
    end

    -- extract vocabulary
    local vocab = {}
    if settings.filter_type ~= nil then
        vocab.train_filter_index = train_filter_index
        vocab.test_filter_index = test_filter_index
    end
    vocab.image_to_index, vocab.index_to_image = 
        util.extract_vocab(util.flatten({train_all_images, test_all_images}))
    vocab.word_to_index, vocab.index_to_word = 
        util.extract_vocab(util.flatten({train_questions, captions}))
    vocab.answer_to_index, vocab.index_to_answer = 
        util.extract_vocab(train_answers)
    vocab['type_to_index'], vocab['index_to_type'] = 
        TYPE_TO_INDEX, INDEX_TO_TYPE

    if settings.discard_unk_answer then
        local images, questions, answers, types = {}, {}, {}, {}
        local answer2index = vocab.answer_to_index
        for i, a in ipairs(test_answers) do
            if answer2index[a] ~= nil then
                table.insert(images, test_images[i])
                table.insert(questions, test_questions[i])
                table.insert(answers, test_answers[i])
                table.insert(types, test_types[i])
            end
        end
        test_images = images
        test_questions = questions
        test_answers = answers
        test_types = types
    end
            
    if settings.top_word then
        -- build word_and_count table
        local word_and_count = {}
        for index, word in ipairs(vocab.index_to_word) do
            word_and_count[word] = 0
        end
        -- count words
        for i, q in ipairs(train_questions) do
            for j, w in ipairs(q) do
                word_and_count[w] = word_and_count[w] + 1
            end
        end
        -- sort by count
        local word_by_count = {}
        for word, count in pairs(word_and_count) do
            table.insert(word_by_count, {word,count})
        end
        table.sort(word_by_count, function(a,b)
                                    return a[2] > b[2]
                                  end)
        local index_to_word, word_to_index = {}, {}
        for i=1,settings.top_word do
            index_to_word[i] = word_by_count[i][1]
            word_to_index[word_by_count[i][1]] = i
        end
        vocab.index_to_word = index_to_word
        vocab.word_to_index = word_to_index
    end

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
    util.index_data(train_questions, vocab.word_to_index, UNK_WORD)
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

    local train_captions, test_captions = nil, nil
    if settings.load_caption then
        local gap
        if settings.load_caption == 'origin' then
            gap = 5
        elseif settings.load_caption == 'generate' then
            gap = 1
        end
            
        -- align captions to images
        local cap_ = {}
        for i = 1,#captions,gap do
            _cap_tuple = {}
            for j=0,gap-1 do
                table.insert(_cap_tuple, captions[i+j])
            end
            table.insert(cap_, _cap_tuple)
        end
        assert(#cap_ == #vocab.index_to_image)
        captions = cap_

        -- duplicat captions to make captions match questions
        train_captions = dup_cap(captions, train_images)
        assert(#train_captions == #train_questions)
        test_captions = dup_cap(captions, test_images)
        assert(#test_captions == #test_questions)
    end

    -- constructure dataset
    local trainset = {
        images = train_images,
        questions = train_questions,
        answers = train_answers, 
        types = train_types,
        captions = train_captions}
    local testset = {
        images = test_images, 
        questions = test_questions,
        answers = test_answers,
        types = test_types,
        captions = test_captions}

    -- add statistic of the dataset
    add_statistic(trainset, vocab)
    add_statistic(testset, vocab)

    -- tf-idf
    if settings.tfidf then
        -- idf
        local idf = torch.zeros(#vocab.index_to_word)
        local function set(alist)
            local _set = {}
            for _, elem in ipairs(alist) do
                _set[elem] = 0
            end
            return _set
        end
        for _, q in ipairs(trainset.questions) do
            for w, _ in pairs(set(q)) do
                idf[w] = idf[w] + 1
            end
        end
        idf = (idf:cinv()*trainset.nsample):log()
        
        -- tf
        local function get_tfidf(dataset)
            local tfidfs = {}
            for i, q in ipairs(dataset.questions) do
                local tf = set(q)
                for _, w in ipairs(q) do
                    tf[w] = tf[w] + 1
                end
                for w, c in pairs(tf) do
                    tf[w] = c/#q
                end
                local tfidf = {}
                for j, w in ipairs(q) do
                    tfidf[j] = --[[tf[w]*]] idf[w]  -- temporarily remove tf[w] for debug
                end
                tfidfs[i] = tfidf
            end
            return tfidfs
        end
        trainset.tfidfs = get_tfidf(trainset)
        testset.tfidfs = get_tfidf(testset)
        vocab.idf = idf
    end

    if settings.bow then
        assert(not settings.tfidf) -- the code of this two part are not compatible
        local function to_bow(dataset)
            local questions = dataset.questions
            for i, q in ipairs(questions) do
                local count = {}
                for j, w in ipairs(q) do
                    if not count[w] then
                        count[w] = 0
                    end
                    count[w] = count[w] + 1
                end
                local bow = {}
                for w, c in pairs(count) do
                    table.insert(bow, {w, c})
                end
                questions[i] = bow
            end
        end
        to_bow(trainset)
        to_bow(testset)
    end

    -- formatting
    if settings.format == 'tensor' then
        format_tensor(trainset)
        format_tensor(testset)
    end

    return trainset, testset, vocab
end

return COCOQA
