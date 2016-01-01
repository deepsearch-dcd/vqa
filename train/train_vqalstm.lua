if vqalstm==nil then
  require('..')
end

-- read command line arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training script for VQA on the DAQUAR dataset.')
cmd:text()
cmd:text('Options')
cmd:option('-model','lstm','Model architecture: [lstm, bilstm, rlstm, rnn, rnnsu, bow]')
cmd:option('-layers',1,'Number of layers (ignored for Tree-LSTM)')
cmd:option('-dim',150,'LSTM memory dimension')
cmd:option('-im_fea_dim',1000,'image feature dimension')
cmd:option('-epochs',50,'Number of training epochs')
cmd:option('-cuda',false,'Using cuda')
cmd:option('-textonly',false,'Text only')
cmd:option('-rmdeter',false,'Remove determiner')
cmd:option('-caption',false,'Use caption')
cmd:option('-dataset','DAQUAR','Dataset [DAQUAR, COCOQA]')
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
local model_class = vqalstm.LSTMVQA
if textonly then
  cmd:log(paths.thisfile() ..'-'.. model_structure .. os.date('_textonly-%Y-%m-%dT%H%M%S') ..'.log')
  header('LSTM for VQA with text only')
else
  cmd:log(paths.thisfile() ..'-'.. model_structure .. os.date('-%Y-%m-%dT%H%M%S') ..'.log')
  header('LSTM for VQA')
end

---------- load dataset ----------
print('loading '.. dataset ..' datasets')
local trainset, testset, vocab
if dataset == 'DAQUAR' then
  trainset, testset, vocab = DAQUAR.process_to_table()
elseif dataset == 'COCOQA' then
  trainset, testset, vocab = COCOQA.load_data{format='table', add_pad_word=false, add_unk_word=true, add_unk_answer=false, load_caption=use_caption}
  trainset.answers = torch.Tensor(trainset.answers)
  testset.answers = torch.Tensor(testset.answers)
else
  error('Unknown dataset')
end

-- [[
if use_caption and textonly then
  for i=1,trainset.size do
    local captions = trainset.captions[i]
    assert(captions~=nil,'caption nil in: '..i)
    local newques = {}
    for j=1,1 do --1,#captions
      local cap = captions[j]
      for k=1,#cap do
        table.insert(newques, cap[k]) --newques[#newques+1] = cap[k]
      end
    end
    local ques = trainset.questions[i]
    for j=1,#ques do
      table.insert(newques, ques[j]) --newques[#newques+1] = ques[j]
    end
    trainset.questions[i] = newques
  end
  trainset.captions = nil
  collectgarbage()
  for i=1,testset.size do
    local captions = testset.captions[i]
    assert(captions~=nil,'caption nil in: '..i)
    local newques = {}
    for j=1,1 do --1,#captions
      local cap = captions[j]
      for k=1,#cap do
        table.insert(newques, cap[k]) --newques[#newques+1] = cap[k]
      end
    end
    local ques = testset.questions[i]
    for j=1,#ques do
      table.insert(newques, ques[j]) --newques[#newques+1] = ques[j]
    end
    testset.questions[i] = newques
  end
  testset.captions = nil
  collectgarbage()

  print('Append captions with question done.')
end
--]]

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
  --addToSet(determiner,vocab.word_to_index['such'])
  --addToSet(determiner,vocab.word_to_index['my'])
  --addToSet(determiner,vocab.word_to_index['your'])
  --addToSet(determiner,vocab.word_to_index['his'])
  --addToSet(determiner,vocab.word_to_index['her'])
  --addToSet(determiner,vocab.word_to_index['our'])
  --addToSet(determiner,vocab.word_to_index['their'])
  --addToSet(determiner,vocab.word_to_index['its'])
  --addToSet(determiner,vocab.word_to_index['some'])
  --addToSet(determiner,vocab.word_to_index['any'])
  --addToSet(determiner,vocab.word_to_index['each'])
  --addToSet(determiner,vocab.word_to_index['every'])
  --addToSet(determiner,vocab.word_to_index['no'])
  --addToSet(determiner,vocab.word_to_index['either'])
  --addToSet(determiner,vocab.word_to_index['neither'])
  --addToSet(determiner,vocab.word_to_index['enough'])
  --addToSet(determiner,vocab.word_to_index['all'])
  --addToSet(determiner,vocab.word_to_index['both'])
  --addToSet(determiner,vocab.word_to_index['several'])
  --addToSet(determiner,vocab.word_to_index['many'])
  --addToSet(determiner,vocab.word_to_index['much'])
  --addToSet(determiner,vocab.word_to_index['few'])
  --addToSet(determiner,vocab.word_to_index['little'])
  --addToSet(determiner,vocab.word_to_index['other'])
  --addToSet(determiner,vocab.word_to_index['another'])

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
    if args.im_fea_dim==1000 then
      feas = npy4th.loadnpy('./feature/COCO-QA/GooLeNet-1000-softmax.npy')
    elseif args.im_fea_dim==1024 then
      feas = npy4th.loadnpy('./feature/COCO-QA/GooLeNet-1024.npy')
    elseif args.im_fea_dim==4096 then
      feas = npy4th.loadnpy('./feature/COCO-QA/VGG19-4096-relu.npy')
    end
  end
  if cuda then
    feas = feas:float():cuda()
  end
  trainset.imagefeas = feas
  testset.imagefeas = feas
end
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