if vqalstm==nil then
  require('..')
end

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

cmd = torch.CmdLine()
cmd:log(paths.thisfile().. '.log')

local args = {}
args.model = 'lstm'
args.layers = 1
args.dim = 150
args.epochs = 50
local usecuda = false
local vocab_size = 10000
local emb_dim = 50
local model_class, model_structure = vqalstm.LSTMVQA, args.model
local num_epochs = args.epochs
header('LSTM for VQA')

---------- load dataset ----------
print('loading datasets')
local trainset, testset, vocab = DAQUAR.process_to_table()
printf('num train = %d\n', trainset.size)
printf('num test  = %d\n', testset.size)
if usecuda then -- data cuda
  trainset.answers = trainset.answers:float():cuda()
  trainset.images = trainset.images:float():cuda()
  testset.answers = testset.answers:float():cuda()
  testset.images = testset.images:float():cuda()
end
---------- load wordvec ----------
local vecs = torch.rand(vocab_size, emb_dim)

---------- initialize model ----------
local model = model_class{
  emb_vecs = vecs,
  structure = model_structure,
  num_layers = args.layers,
  mem_dim = args.dim,
  num_classes = trainset.nanswer,
  cuda = usecuda
}

---------- print information ----------
header('model configuration')
print('max epochs = '.. num_epochs) -- printf('max epochs = %d\n', num_epochs)
model:print_config()

---------- TRAIN ----------
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  print('-- epoch '.. i) -- printf('-- epoch %d\n', i)
  model:train(trainset)
  print('-- finished epoch in '.. (sys.clock() - start) .. 's') -- printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  
  -- uncomment to compute train scores
  -- [[
  local train_predictions = model:predict_dataset(trainset)
  local train_score = accuracy(train_predictions, trainset.answers)
  print('-- train score: '.. train_score) -- printf('-- train score: %.4f\n', train_score)
  --]]

  local dev_predictions = model:predict_dataset(testset)
  local dev_score = accuracy(dev_predictions, testset.answers)
  print('-- test score: '.. dev_score) -- printf('-- test score: %.4f\n', dev_score)

  if dev_score > best_dev_score then
    best_dev_score = dev_score
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      num_layers = args.layers,
      mem_dim = args.dim,
      num_classes = trainset.nanswer,
      cuda = usecuda
    }
    best_dev_model.params:copy(model.params)
    best_dev_model.emb.weight:copy(model.emb.weight)
  end
end
print('finished training in '.. (sys.clock() - train_start) ..'s') -- printf('finished training in %.2fs\n', sys.clock() - train_start)
