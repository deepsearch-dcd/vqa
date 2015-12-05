if vqalstm==nil then
  require('..')
end

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training script for VQA on the DAQUAR dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-mpath','done/vqalstm_textonly-rnn.l1.d150.e37.c1-2015-12-04T081911.t7','Model path')
cmd:text()
local args = cmd:parse(arg)

---------- Load model ----------
local model_save_path = args.mpath
local cuda = string.find(model_save_path, 'c1') and true or false
local textonly = string.find(model_save_path, 'textonly') and true or false
local model_class
if textonly then
  model_class = vqalstm.LSTMVQATO
  header('LSTM for VQA with text only')
else
  model_class = vqalstm.LSTMVQA
  header('LSTM for VQA')
end

local model = model_class.load(model_save_path)

---------- load dataset ----------
print('loading datasets')
local trainset, testset, vocab = DAQUAR.process_to_table()
for i=1,trainset.size do
  trainset.questions[i] = torch.Tensor(trainset.questions[i])
end
for i=1,testset.size do
  testset.questions[i] = torch.Tensor(testset.questions[i])
end

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
  feas = npy4th.loadnpy('./feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')
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
local testnum = 20
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
