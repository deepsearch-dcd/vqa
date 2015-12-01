if vqalstm==nil then
  require('..')
end

cmd = torch.CmdLine()
cmd:log(paths.thisfile() .. os.date('-%Y-%m-%dT%H%M%S') ..'.log')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

-- read command line arguments
local args = lapp [[
Training script for sentiment classification on the SST dataset.
  -m,--model  (default lstm)    Model architecture: [lstm, bilstm]
  -l,--layers (default 1)       Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)     LSTM memory dimension
  -e,--epochs (default 50)      Number of training epochs
  -c,--cuda   (default true)    Using cuda
]]

--[[
local args = {}
args.model = 'lstm'
args.layers = 1
args.dim = 150
args.epochs = 50
args.cuda = true
--]]
local vocab_size = 10000
local emb_dim = 50
local model_class, model_structure = vqalstm.LSTMVQA, args.model
local num_epochs = args.epochs
local cuda = args.cuda
header('LSTM for VQA')

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
print('loading features')
feas = npy4th.loadnpy('./feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')
if cuda then
  feas = feas:float():cuda()
end
trainset.imagefeas = feas
testset.imagefeas = feas

---------- load wordvec ----------
local vecs = torch.rand(vocab_size, emb_dim)

---------- initialize model ----------
local model = model_class{
  emb_vecs = vecs,
  structure = model_structure,
  num_layers = args.layers,
  mem_dim = args.dim,
  num_classes = trainset.nanswer,
  cuda = args.cuda,
  im_fea_dim = args.im_fea_dim
}

---------- print information ----------
header('model configuration')
print(string.format('%-25s = %d',   'max epochs', num_epochs))
model:print_config()

---------- TRAIN ----------
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
local best_dev_epoch = 1
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  print('-- epoch '.. i)
  model:train(trainset)
  print('-- finished epoch in '.. string.format("%.2fs", (sys.clock() - start)))
  
  -- uncomment to compute train scores
  -- [[
  start = sys.clock()
  local train_predictions = model:predict_dataset(trainset)
  local train_score = accuracy(train_predictions, trainset.answers)
  print('-- train score: '.. train_score ..', cost '.. string.format("%.2fs", (sys.clock() - start)))
  --]]

  start = sys.clock()
  local dev_predictions = model:predict_dataset(testset)
  local dev_score = accuracy(dev_predictions, testset.answers)
  print('-- test score: '.. dev_score ..', cost '.. string.format("%.2fs", (sys.clock() - start)))

  if dev_score > best_dev_score then
    best_dev_score = dev_score
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      num_layers = args.layers,
      mem_dim = args.dim,
      num_classes = trainset.nanswer,
      cuda = args.cuda,
      im_fea_dim = args.im_fea_dim
    }
    best_dev_model.params:copy(model.params)
    best_dev_model.emb.weight:copy(model.emb.weight)
    best_dev_epoch = i
  end
end
print('finished training in '.. string.format("%.2fs", (sys.clock() - train_start)))

---------- Save model ----------
local model_save_path = string.format("./done/vqalstm-%s.l%d.d%d.e%d.c%d-%s.t7", args.model, args.layers, args.dim, best_dev_epoch, args.cuda and 1 or 0, os.date('%Y-%m-%dT%H%M%S'))

-- write model to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

-- to load a saved model
--local loaded_model = model_class.load(model_save_path)