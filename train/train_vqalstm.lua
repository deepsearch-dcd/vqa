if vqalstm==nil then
  require('..')
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training script for VQA on VQA dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-model','lstm','Model architecture: [lstm, bilstm, rlstm, rnn, rnnsu, bow]')
cmd:option('-layers',1,'Number of layers (ignored for Tree-LSTM)')
cmd:option('-dim',150,'LSTM memory dimension')
cmd:option('-im_fea_dim',1024,'image feature dimension')
cmd:option('-epochs',100,'Number of training epochs')
cmd:option('-cuda',false,'Using cuda')
cmd:option('-textonly',false,'Text only')
cmd:option('-rmdeter',false,'Remove determiner')
cmd:option('-caption',false,'Use caption')
cmd:option('-dataset','COCOQA','Dataset [DAQUAR, COCOQA]')
cmd:option('-modelclass','LSTMVQA','Model class [LSTMVQA, ConcatVQA]')
cmd:text()
local args = cmd:parse(arg)

--[[
local args = {}
args.model = 'lstm'
args.layers = 1
args.dim = 150
args.epochs = 50
args.cuda = true
args.textonly = true
--]]
local emb_dim = 50
local model_structure = args.model
local num_epochs = args.epochs
local cuda = args.cuda
local textonly = args.textonly
local dataset = args.dataset
local use_caption = args.caption
local model_class
if args.modelclass == 'LSTMVQA' then
  model_class = vqalstm.LSTMVQA
elseif args.modelclass == 'ConcatVQA' then
  model_class = vqalstm.ConcatVQA
else
  error('Unknown model class')
end
if textonly then
  cmd:log(paths.thisfile() ..'-'.. model_structure .. os.date('_textonly-%Y-%m-%dT%H%M%S') ..'.log')
  header('LSTM for VQA with text only')
else
  cmd:log(paths.thisfile() ..'-'.. model_structure .. os.date('-%Y-%m-%dT%H%M%S') ..'.log')
  header('LSTM for VQA')
end

---------- load dataset ----------
local trainset, testset, vocab = loadData(args)
print('num train = '.. trainset.size)
print('num test  = '.. testset.size)

---------- load wordvec ----------
local vecs = torch.rand(trainset.nvocab, emb_dim)

---------- initialize model ----------
local model = model_class{
  emb_vecs = vecs,
  structure = model_structure,
  num_layers = args.layers,
  mem_dim = args.dim,
  num_classes = trainset.nanswer,
  cuda = args.cuda,
  im_fea_dim = args.im_fea_dim,
  textonly = textonly
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
      im_fea_dim = args.im_fea_dim,
      textonly = textonly
    }
    best_dev_model.params:copy(model.params)
    best_dev_model.emb.weight:copy(model.emb.weight)
    best_dev_epoch = i
  end
end
print('finished training in '.. string.format("%.2fs", (sys.clock() - train_start)))
print('best dev score is: '.. best_dev_score)

---------- Save model ----------
local model_save_path
if textonly then
  model_save_path = string.format("./done/vqalstm-%s-%s_textonly.l%d.d%d.e%d.c%d-%s.t7", 
    args.dataset, args.model, args.layers, args.dim, best_dev_epoch, args.cuda and 1 or 0, 
    os.date('%Y-%m-%dT%H%M%S'))
else
  model_save_path = string.format("./done/vqalstm-%s-%s.l%d.d%d.e%d.c%d-%s.t7", 
    args.dataset, args.model, args.layers, args.dim, best_dev_epoch, args.cuda and 1 or 0, 
    os.date('%Y-%m-%dT%H%M%S'))
end

-- write model to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

-- to load a saved model
--local loaded_model = model_class.load(model_save_path)