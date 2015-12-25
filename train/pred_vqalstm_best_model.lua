if vqalstm==nil then
  require('..')
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training script for VQA on the DAQUAR dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-mpath','done/vqalstm-DAQUAR-rnn_textonly.l1.d150.e37.c1-2015-12-04T081911.t7','Model path')
cmd:option('-testnum',20,'Test sample number')
cmd:option('-rmdeter',false,'Remove determiner')
cmd:text()
local args = cmd:parse(arg)

---------- Load model ----------
local testnum = args.testnum
local model_save_path = args.mpath
local cuda = string.find(model_save_path, 'c1') and true or false
local textonly = string.find(model_save_path, 'textonly') and true or false
local dataset
if string.find(model_save_path, 'DAQUAR') then
  dataset = 'DAQUAR'
elseif string.find(model_save_path, 'COCOQA') then
  dataset = 'COCOQA'
else
  dataset = 'DAQUAR'
end
local model_class = vqalstm.LSTMVQA
if textonly then
  header('LSTM for VQA with text only')
else
  header('LSTM for VQA')
end

local model = model_class.load(model_save_path)

-- write model to disk
--print('writing model to ' .. model_save_path)
--model:save(model_save_path)

---------- load dataset ----------
print('loading '.. dataset ..' datasets')
local trainset, testset, vocab
if dataset == 'DAQUAR' then
  trainset, testset, vocab = DAQUAR.process_to_table()
elseif dataset == 'COCOQA' then
  trainset, testset, vocab = COCOQA.load_data{format='table', add_pad_word=false, add_unk_word=true, add_unk_answer=false}
  trainset.answers = torch.Tensor(trainset.answers)
  testset.answers = torch.Tensor(testset.answers)
else
  error('Unknown dataset')
end

-- Remove determiner
if args.rmdeter then
  -- build determiner set
  local determiner = {}
  addToSet(determiner,vocab.word_to_index['a'])
  addToSet(determiner,vocab.word_to_index['an'])
  addToSet(determiner,vocab.word_to_index['the'])
  addToSet(determiner,vocab.word_to_index['this'])
  addToSet(determiner,vocab.word_to_index['that'])
  addToSet(determiner,vocab.word_to_index['these'])
  addToSet(determiner,vocab.word_to_index['those'])
  addToSet(determiner,vocab.word_to_index['such'])
  addToSet(determiner,vocab.word_to_index['my'])
  addToSet(determiner,vocab.word_to_index['your'])
  addToSet(determiner,vocab.word_to_index['his'])
  addToSet(determiner,vocab.word_to_index['her'])
  addToSet(determiner,vocab.word_to_index['our'])
  addToSet(determiner,vocab.word_to_index['their'])
  addToSet(determiner,vocab.word_to_index['its'])
  addToSet(determiner,vocab.word_to_index['some'])
  addToSet(determiner,vocab.word_to_index['any'])
  addToSet(determiner,vocab.word_to_index['each'])
  addToSet(determiner,vocab.word_to_index['every'])
  --addToSet(determiner,vocab.word_to_index['no'])
  addToSet(determiner,vocab.word_to_index['either'])
  addToSet(determiner,vocab.word_to_index['neither'])
  addToSet(determiner,vocab.word_to_index['enough'])
  addToSet(determiner,vocab.word_to_index['all'])
  addToSet(determiner,vocab.word_to_index['both'])
  addToSet(determiner,vocab.word_to_index['several'])
  addToSet(determiner,vocab.word_to_index['many'])
  addToSet(determiner,vocab.word_to_index['much'])
  addToSet(determiner,vocab.word_to_index['few'])
  --addToSet(determiner,vocab.word_to_index['little'])
  addToSet(determiner,vocab.word_to_index['other'])
  addToSet(determiner,vocab.word_to_index['another'])

  for i=1,trainset.size do
    local ques = trainset.questions[i]
    for j=#ques,1,-1 do
      if setContains(determiner, ques[j]) then
        table.remove(ques,j)
      end
    end
  end
  for i=1,testset.size do
    local ques = testset.questions[i]
    for j=#ques,1,-1 do
      if setContains(determiner, ques[j]) then
        table.remove(ques,j)
      end
    end
  end

  print('Remove determiner done.')
end

-- convert table to Tensor
for i=1,trainset.size do
  trainset.questions[i] = torch.Tensor(trainset.questions[i])
end
for i=1,testset.size do
  testset.questions[i] = torch.Tensor(testset.questions[i])
end

-- convert to cuda
if cuda then
  for i=1,trainset.size do
    trainset.questions[i] = trainset.questions[i]:float():cuda()
  end
  for i=1,testset.size do
    testset.questions[i] = testset.questions[i]:float():cuda()
  end
end

print('num train = '.. trainset.size)
print('num test  = '.. testset.size)

---------- load features ----------
if not textonly then
  print('loading features')
  local feas
  if dataset == 'DAQUAR' then
    feas = npy4th.loadnpy('./feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')
  elseif dataset == 'COCOQA' then
    feas = npy4th.loadnpy('./feature/COCO-QA/GooLeNet-1000-softmax.npy')
  end
  if cuda then
    feas = feas:float():cuda()
  end
  trainset.imagefeas = feas
  testset.imagefeas = feas
end

---------- print information ----------
header('model configuration')
model:print_config()

---------- PREDICT ----------
header('Prediction')
start = sys.clock()
local train_predictions = model:predict_dataset(trainset)
local train_score = accuracy(train_predictions, trainset.answers)
print('-- train score: '.. train_score ..', cost '.. string.format("%.2fs", (sys.clock() - start)))

start = sys.clock()
local dev_predictions = model:predict_dataset(testset)
local dev_score = accuracy(dev_predictions, testset.answers)
print('-- test score: '.. dev_score ..', cost '.. string.format("%.2fs", (sys.clock() - start)))

---------- Print Samples ----------
print('===== Training Samples =====')
for k=1,testnum do
  local i = math.random(trainset.size)
  local que = trainset.questions[i]
  local ans = trainset.answers[i]
  local pre = train_predictions[i]
  local ques = {}
  for j=1,que:size(1) do
    table.insert(ques, vocab.index_to_word[que[j]])
  end
  print('Ques:'.. string.format(string.rep(' %s',que:size(1)), unpack(ques)))
  print('--Answ: '.. vocab.index_to_answer[ans])
  print('--Pred: '.. vocab.index_to_answer[pre])
end

print('===== Testing Samples =====')
for k=1,testnum do
  local i = math.random(testset.size)
  local que = testset.questions[i]
  local ans = testset.answers[i]
  local pre = train_predictions[i]
  local ques = {}
  for j=1,que:size(1) do
    table.insert(ques, vocab.index_to_word[que[j]])
  end
  print('Ques:'.. string.format(string.rep(' %s',que:size(1)), unpack(ques)))
  print('--Answ: '.. vocab.index_to_answer[ans])
  print('--Pred: '.. vocab.index_to_answer[pre])
end
