if vqalstm==nil then
  require('..')
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Prediction script for VQA on VQA dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-mpath','done/vqalstm-DAQUAR-rnn_textonly.l1.d150.e37.c1-2015-12-04T081911.t7','Model path')
cmd:option('-testnum',20,'Test sample number')
cmd:option('-im_fea_dim',1024,'image feature dimension')
cmd:option('-rmdeter',false,'Remove determiner')
cmd:option('-caption',false,'Use caption')
cmd:text()
local args = cmd:parse(arg)

---------- Load model ----------
local testnum = args.testnum
local model_save_path = args.mpath
args.cuda = string.find(model_save_path, 'c1') and true or false
args.textonly = string.find(model_save_path, 'textonly') and true or false
if string.find(model_save_path, 'DAQUAR') then
  args.dataset = 'DAQUAR'
elseif string.find(model_save_path, 'COCOQA') then
  args.dataset = 'COCOQA'
else
  args.dataset = 'DAQUAR'
end
local model_class = vqalstm.LSTMVQA
if args.textonly then
  header('LSTM for VQA with text only')
else
  header('LSTM for VQA')
end

local model = model_class.load(model_save_path)

-- write model to disk
--print('writing model to ' .. model_save_path)
--model:save(model_save_path)

---------- load dataset ----------
local trainset, testset, vocab = loadData(args)
print('num train = '.. trainset.size)
print('num test  = '.. testset.size)

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
