require 'paths'
local torch = require 'torch'

util = require 'util/util'

local DAQUAR = {}

local unk, pad = '*unk*', '*pad*'

--[[
Process the raw data file
@return
images: [table of string] the i-th entry is the image name associated with the i-th question.
questions: [table of tables] the i-th subtable contains all the words of the i-th question.
answers: [table of string] the i-th entry is the answer associated with the i-th question.
--]]
local function process_raw(file_path)
	-- read dataset file
	local f = assert(io.open(file_path, 'r'))
	local file_content = f:read('*all')
	f:close()

	local lines = util.split_line(file_content)
	print('total lines: '..#lines)
	assert(#lines % 2 == 0)
	
	local questions, answers, images = {}, {}, {}
	local count = 0
	-- odd-th line is question and even-th line is answer.
	for _, l in ipairs(lines) do
		count = count + 1
		if count % 2 == 1 then
			table.insert(questions, util.split_word(l))
		else
			table.insert(answers, l)
		end
	end
	assert(#questions == #answers)

	-- get the image list from questions and count the max length by the way
	local max_length, min_length = 0, 100
	for _, q in ipairs(questions) do
		if #q > max_length then
			max_length = #q
		end
		if #q < min_length then
			min_length = #q
		end
		i = q[#q-1]
		assert(string.find(i, '^image%d+$'))
		table.insert(images, i)
	end
	print('max length: '..max_length..' min length: '..min_length)
	assert(#questions == #images)
	print('#images: '..#images..' #questions: '..#questions..' #answers: '..#answers)

	-- normalize questions, e.g. switch 'image123' to 'image'
	count = 0
	for i, q in ipairs(questions) do
		for j , w in ipairs(q) do
			if string.find(w, '^image%d+$') then
				questions[i][j] = 'image'
				count = count + 1
			end
		end
	end
	print('normalize image: '..count)

	return images, questions, answers
end

local function extract_word_vocab(questions)
	return util.extract_vocab(util.flatten(questions), pad, unk)
end

local function extract_answer_vocab(answers)
	return util.extract_vocab(answers, unk)
end

local function extract_image_vocab(train_list, test_list)
	return util.extract_vocab(util.flatten({train_list, test_list}))
end
	
--[[
Convert words in questions to index according to the given vocabulary.
Assume the padding word comes first in the vocabulary.
@param
length: all questions will aligned to this length, less padding, more truncate.
@return
data: [Tensor] each row is a question.
--]]
local function questions_to_index(questions, word_to_index, length)
	local data = torch.Tensor(#questions, length):fill(word_to_index[pad])
	for i, q in ipairs(questions) do
		for j, w in ipairs(q) do
			if j > length then break end
			if word_to_index[w] then
				data[i][j] = word_to_index[w]
			else
				data[i][j] = word_to_index[unk]
			end
		end
	end
	return data
end

local function answers_to_index(answers, answer_to_index)
	local data = torch.Tensor(#answers)
	for i, a in ipairs(answers) do
		if answer_to_index[a] then
			data[i] = answer_to_index[a]
		else
			data[i] = answer_to_index[unk]
		end
	end
	return data
end

local function images_to_index(images, image_to_index)
	local data = torch.Tensor(#images)
	for i, m in ipairs(images) do
		data[i] = image_to_index[m]
	end
	return data
end

local function to_index(images, questions, answers, vocab, length)
	local dataset = {}
	dataset.images = images_to_index(images, vocab.image_to_index)
	dataset.questions = questions_to_index(questions, 
						vocab.word_to_index, 
						length)
	dataset.answers = answers_to_index(answers, vocab.answer_to_index)

    -- some statistics
    dataset.nimage = #vocab.index_to_image  -- total distinct images
    dataset.nvocab = #vocab.index_to_word   -- total distinct words
    dataset.nanswer = #vocab.index_to_answer    -- total distinct answer
    dataset.nsample = dataset.images:size(1)

    --return util.index_dataset(dataset)
    return dataset
end

local function dump_image_list(dest_path, index_to_image)
	print('dump image list to: '..dest_path)
	local f = assert(io.open(dest_path, 'w'))
	for _, i in ipairs(index_to_image) do
		f:write(i..'\n')
	end
	f:close()
end

local function process_all(raw_train_path, raw_test_path, trainset_path, testset_path,image_list_path, vocab_path)
	print('load train set...')
	train_images, train_questions, train_answers = process_raw(raw_train_path)
	print('load test set...')
	test_images, test_questions, test_answers = process_raw(raw_test_path)
	
	-- build vocab
	print('build vocabulary...')
	local vocab = {}
	vocab['word_to_index'], vocab['index_to_word'] = 
		extract_word_vocab(train_questions)
	print('word vocabulary: '..#vocab['index_to_word'])
	vocab['answer_to_index'], vocab['index_to_answer'] =
		extract_answer_vocab(train_answers)
	print('answer vocabulary: '..#vocab['index_to_answer'])
	vocab['image_to_index'], vocab['index_to_image'] =
		extract_image_vocab(train_images, test_images)
	print('image vocabulary: '..#vocab['index_to_image'])
	dump_image_list(image_list_path, vocab['index_to_image'])
	torch.save(vocab_path, vocab)

	-- convert to index
	print('save trainset...')
	trainset = to_index(train_images, train_questions, train_answers, vocab, 30)
	torch.save(trainset_path, trainset)
	print('save testset...')
	testset = to_index(test_images, test_questions, test_answers, vocab, 30)
	torch.save(testset_path, testset)
	
	return trainset, testset, vocab
end

local function back_to_question(stream, index_to_word, word_to_index)
	local question = {}
	for i = 1, stream:nElement() do
		if stream[i] == word_to_index[pad] then break end
		table.insert(question, index_to_word[stream[i]])
	end
	return question
end

local function check_dataset(before_path, after_path, vocab)
	local dataset = torch.load(after_path)
	assert(dataset['images']:size(1) == dataset['questions']:size(1))
	assert(dataset['questions']:size(1) == dataset['answers']:size(1))
	local images, questions, answers = {}, {}, {}
	
	for i = 1, dataset['images']:size(1) do
		table.insert(images, vocab['index_to_image'][dataset['images'][i]])
	end

	for i = 1, dataset['questions']:size(1) do
		table.insert(questions, back_to_question(dataset['questions'][i], vocab['index_to_word'], vocab['word_to_index']))
	end
	assert(#images == #questions)

	for i = 1, dataset['answers']:size(1) do
		table.insert(answers, vocab['index_to_answer'][dataset['answers'][i]])
	end
	assert(#questions == #answers)

	b_images, b_questions, b_answers = process_raw(before_path)
	assert(images, b_images)
	assert(#questions, #b_questions)
	assert(#answers, #b_answers)

	for i, q in ipairs(questions) do
		for j, w in ipairs(q) do
			if w ~= unk and w ~= 'image' then
				assert(w == b_questions[i][j])
			end
		end
	end

	for i, a in ipairs(answers) do
		if a ~= unk then
			assert(a == b_answers[i])
		end
	end
end

local function check_all(raw_train_path, raw_test_path, trainset_path, testset_path, vocab_path)
	
	-- load vocabulary
	vocab = torch.load(vocab_path)

	print('check trainset...')
	check_dataset(raw_train_path, trainset_path, vocab)
		
	print('check testset...')
	check_dataset(raw_test_path, testset_path, vocab)
end	

function DAQUAR.process_and_check()
	local data_dir = paths.dirname(paths.thisfile())
	data_dir = paths.concat(data_dir, 'data/DAQUAR/DAQUAR-ALL')
	done_dir = paths.concat(data_dir, 'done')
	paths.mkdir(done_dir)

	local raw_train_path = paths.concat(data_dir, 'qa.894.raw.train.txt')
	local raw_test_path = paths.concat(data_dir, 'qa.894.raw.test.txt')
	local trainset_path = paths.concat(done_dir, 'trainset.t7')
	local testset_path = paths.concat(done_dir, 'testset.t7')
	local image_list_path = paths.concat(done_dir, 'image_list.txt')
	local vocab_path = paths.concat(done_dir, 'vocab.t7')

	print('process data at: '..data_dir)
	trainset, testset, vocab = process_all(raw_train_path, raw_test_path, trainset_path, testset_path, image_list_path, vocab_path)

	print('check data at: '..done_dir)
	check_all(raw_train_path, raw_test_path, trainset_path, testset_path, vocab_path)

	return trainset, testset, vocab
end

return DAQUAR
