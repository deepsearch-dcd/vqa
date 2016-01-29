if vqalstm==nil then
  require('..')
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Prediction script for VQA on VQA dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-mpath','','Model path')--'done/vqalstm-COCOQA-lstm_textonly.l1.d150.e20.c1-2016-01-26T133703.t7'
cmd:option('-testnum',20,'Test sample number')

cmd:option('-model','lstm','Model architecture: [lstm, bilstm, rlstm, rnn, rnnsu, bow]')
cmd:option('-layers',1,'Number of layers (ignored for Tree-LSTM)')
cmd:option('-dim',150,'LSTM memory dimension')
cmd:option('-im_fea_dim',1024,'image feature dimension')
cmd:option('-epochs',100,'Number of training epochs')
cmd:option('-cuda',false,'Using cuda')
cmd:option('-textonly',false,'Text only')
cmd:option('-rmdeter',false,'Remove determiner')
cmd:option('-caption',false,'Use caption')
cmd:option('-capopt','origin','Caption option [origin, generate]')
cmd:option('-caponly',false,'Use caption only without question')
cmd:option('-dataset','COCOQA','Dataset [DAQUAR, COCOQA]')
cmd:option('-modelclass','LSTMVQA','Model class [LSTMVQA, ConcatVQA]')
cmd:text()
local args = cmd:parse(arg)

---------- Load model ----------
local testnum = args.testnum
local model_save_path = args.mpath
if string.find(model_save_path, 'lstm') then
  args.model = 'lstm'
elseif string.find(model_save_path, 'bilstm') then
  args.model = 'bilstm'
elseif string.find(model_save_path, 'rlstm') then
  args.model = 'rlstm'
elseif string.find(model_save_path, 'rnn') then
  args.model = 'rnn'
elseif string.find(model_save_path, 'rnnsu') then
  args.model = 'rnnsu'
elseif string.find(model_save_path, 'bow') then
  args.model = 'bow'
else
  args.model = 'lstm'
end
args.layers = string.match(model_save_path, 'l(%d+)')
args.dim = string.match(model_save_path, 'd(%d+)')
args.epochs = string.match(model_save_path, 'e(%d+)')
args.cuda = string.find(model_save_path, 'c1') and true or false
args.textonly = string.find(model_save_path, 'textonly') and true or false
if string.find(model_save_path, 'DAQUAR') then
  args.dataset = 'DAQUAR'
elseif string.find(model_save_path, 'COCOQA') then
  args.dataset = 'COCOQA'
else
  args.dataset = 'DAQUAR'
end
local model_class
if args.modelclass == 'LSTMVQA' then
  model_class = vqalstm.LSTMVQA
elseif args.modelclass == 'ConcatVQA' then
  model_class = vqalstm.ConcatVQA
else
  error('Unknown model class')
end
if args.textonly then
  header('LSTM for VQA with text only')
else
  header('LSTM for VQA')
end
print(cmd:string(paths.thisfile(), args, {dir=true}))

---------- load dataset ----------
local trainset, testset, vocab = loadData(args)
print('num train = '.. trainset.size)
print('num test  = '.. testset.size)

---------- load model ----------
local model = model_class.load(model_save_path)

---------- print information ----------
header('model configuration')
print(string.format('%-25s = %d',   'best epoch', args.epochs))
model:print_config()

-- write model to disk
--print('writing model to ' .. model_save_path)
--model:save(model_save_path)

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

local acc_save_name = string.format("%s-PRED.t7", model_save_path)
torch.save(acc_save_name, {
  train_predictions = train_predictions,
  dev_predictions = dev_predictions,
  train_score = train_score,
  dev_score = dev_score,
  args = args,
  })

args.rmdeter = false
if args.dataset == 'COCOQA' and args.caption then
  trainset, testset, vocab = COCOQA.load_data{format='table', add_pad_word=false, 
    add_unk_word=true, add_unk_answer=false, load_caption=args.capopt}
  trainset.answers = torch.Tensor(trainset.answers)
  testset.answers = torch.Tensor(testset.answers)
end

---------- Print Samples ----------
header('===== Training Samples =====')
for k=1,testnum do
  local i = math.random(trainset.size)
  local que = trainset.questions[i]
  local cap = trainset.captions[i][1]
  local ans = trainset.answers[i]
  local pre = train_predictions[i]
  local ques = {}
  for j=1,#que do
    table.insert(ques, vocab.index_to_word[que[j]])
  end
  local caps = {}
  for j=1,#cap do
    table.insert(caps, vocab.index_to_word[cap[j]])
  end
  print('Question:'.. string.format(string.rep(' %s',#que), unpack(ques)))
  print('Caption:'.. string.format(string.rep(' %s',#cap), unpack(caps)))
  print('--Answer: '.. vocab.index_to_answer[ans])
  print('--Predict: '.. vocab.index_to_answer[pre])
  print('--Picture: '.. 
    string.format('train2014/COCO_train2014_%012d', 
      tonumber(vocab.index_to_image[trainset.images[i]])))
end

header('===== Testing Samples =====')
for k=1,testnum do
  local i = math.random(testset.size)
  local que = testset.questions[i]
  local cap = testset.captions[i][1]
  local ans = testset.answers[i]
  local pre = dev_predictions[i]
  local ques = {}
  for j=1,#que do
    table.insert(ques, vocab.index_to_word[que[j]])
  end
  local caps = {}
  for j=1,#cap do
    table.insert(caps, vocab.index_to_word[cap[j]])
  end
  print('Question:'.. string.format(string.rep(' %s',#que), unpack(ques)))
  print('Caption:'.. string.format(string.rep(' %s',#cap), unpack(caps)))
  print('--Answer: '.. vocab.index_to_answer[ans])
  print('--Predict: '.. vocab.index_to_answer[pre])
  print('--Picture: '.. 
    string.format('train2014/COCO_train2014_%012d', 
      tonumber(vocab.index_to_image[testset.images[i]])))
end
