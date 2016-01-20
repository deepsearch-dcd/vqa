if vqalstm==nil then
  require('..')
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Continue training script for VQA on VQA dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-mpath','done/vqalstm-COCOQA-bow_textonly.l1.d150.e38.c1-2016-01-01T185418.t7','Model path')
cmd:option('-im_fea_dim',1024,'image feature dimension')
cmd:option('-epochs',100,'Number of training epochs')
cmd:option('-rmdeter',false,'Remove determiner')
cmd:option('-caption',false,'Use caption')
cmd:text()
local args = cmd:parse(arg)

---------- Load args ----------
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
  error('Unknown model')
end
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
  cmd:log(paths.thisfile() ..'-'.. args.model .. os.date('_textonly-%Y-%m-%dT%H%M%S') ..'.log')
  header('LSTM Continue for VQA with text only')
else
  cmd:log(paths.thisfile() ..'-'.. args.model .. os.date('-%Y-%m-%dT%H%M%S') ..'.log')
  header('LSTM Continue for VQA')
end

---------- load dataset ----------
local trainset, testset, vocab = loadData(args)
print('num train = '.. trainset.size)
print('num test  = '.. testset.size)

---------- load model ----------
local model = model_class.load(model_save_path)

---------- print information ----------
header('model configuration')
print(string.format('%-25s = %d',   'max epochs', args.epochs))
model:print_config()

---------- TRAIN ----------
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
local best_dev_epoch = tonumber(string.match(model_save_path,'e(%d+)'))

print('-- best dev epoch '.. best_dev_epoch)
start = sys.clock()
local train_predictions = model:predict_dataset(trainset)
local train_score = accuracy(train_predictions, trainset.answers)
print('-- train score: '.. train_score ..', cost '.. string.format("%.2fs", (sys.clock() - start)))
start = sys.clock()
local dev_predictions = model:predict_dataset(testset)
local best_dev_score = accuracy(dev_predictions, testset.answers)
print('-- test score: '.. best_dev_score ..', cost '.. string.format("%.2fs", (sys.clock() - start)))

header('Continue Training model')
for i = best_dev_epoch, args.epochs do
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
    --best_dev_model = model_class{
    --  emb_vecs = model.emb.weight:float(),
    --  structure = args.model,
    --  num_layers = model.num_layers,
    --  mem_dim = model.mem_dim,
    --  num_classes = trainset.nanswer,
    --  cuda = args.cuda,
    --  im_fea_dim = args.im_fea_dim,
    --  textonly = args.textonly
    --}
    --best_dev_model.params:copy(model.params)
    --best_dev_model.emb.weight:copy(model.emb.weight)
    best_dev_epoch = i
  end
end
print('finished training in '.. string.format("%.2fs", (sys.clock() - train_start)))
print('best dev score is: '.. best_dev_score)

---------- Save model ----------
local model_save_path
if args.textonly then
  model_save_path = string.format("./done/vqalstm-%s-%s_textonly.l%d.d%d.e%d.c%d-%s.t7", 
    args.dataset, args.model, model.num_layers, model.mem_dim, best_dev_epoch, args.cuda and 1 or 0, 
    os.date('%Y-%m-%dT%H%M%S'))
else
  model_save_path = string.format("./done/vqalstm-%s-%s.l%d.d%d.e%d.c%d-%s.t7", 
    args.dataset, args.model, model.num_layers, model.mem_dim, best_dev_epoch, args.cuda and 1 or 0, 
    os.date('%Y-%m-%dT%H%M%S'))
end

-- write model to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)
