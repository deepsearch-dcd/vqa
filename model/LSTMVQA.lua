--[[

  VQA using LSTMs.

--]]

local LSTMVQA = torch.class('vqalstm.LSTMVQA')

function LSTMVQA:__init(config)
  self.mem_dim           = config.mem_dim           or 150
  self.learning_rate     = config.learning_rate     or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.num_layers        = config.num_layers        or 1
  self.batch_size        = config.batch_size        or 1 -- train per 1 sample
  self.reg               = config.reg               or 1e-4
  self.structure         = config.structure         or 'lstm' -- {lstm, bilstm, rlstm, gru, bigru, rnn, rnnsu, bow}
  self.dropout           = (config.dropout == nil) and true or config.dropout
  self.num_classes       = config.num_classes
  self.cuda              = config.cuda              or false
  self.textonly          = config.textonly          or false
  self.imageonly         = config.imageonly         or false
  self.im_fea_dim        = config.im_fea_dim        or 1000
  assert(self.num_classes~=nil)
  assert((self.textonly and self.imageonly)~=true)
  self.imagetext         = not (self.textonly or self.imageonly) -- use image plus text

  if not self.imageonly then
    -- word embedding
    self.emb_dim = config.emb_vecs:size(2)
    self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
    self.emb.weight:copy(config.emb_vecs)
  end

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  -- vqa classification module
  self.vqa_module = self:new_vqa_module()

  if self.imagetext then
    self.jointable2 = nn.JoinTable(2)
  end

  if self.cuda then
    if not self.imageonly then
      self.emb = self.emb:cuda()
    end
    self.criterion = self.criterion:cuda()
    self.vqa_module = self.vqa_module:cuda()
    if self.imagetext then
      self.jointable2 = self.jointable2:cuda()
    end
  end

  -- initialize LSTM model
  local lstm_config = {
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = true,
    cuda = self.cuda
  }
  if self.textonly then
    lstm_config.in_dim = self.emb_dim
  elseif self.imageonly then
    lstm_config.in_dim = self.im_fea_dim
  else
    lstm_config.in_dim = self.emb_dim+self.im_fea_dim
  end

  if self.structure == 'lstm' or self.structure == 'rlstm' then
    self.lstm = vqalstm.LSTM(lstm_config)
  elseif self.structure == 'bilstm' then
    self.lstm = vqalstm.LSTM(lstm_config)
    self.lstm_b = vqalstm.LSTM(lstm_config)
  elseif self.structure == 'gru' then
    self.lstm = vqalstm.GRU(lstm_config)
  elseif self.structure == 'bigru' then
    self.lstm = vqalstm.GRU(lstm_config)
    self.lstm_b = vqalstm.GRU(lstm_config)
  elseif self.structure == 'rnn' then
    self.lstm = vqalstm.RNN(lstm_config)
  elseif self.structure == 'rnnsu' then
    self.lstm = vqalstm.RNNSU(lstm_config)
  elseif self.structure == 'bow' then
    self.lstm = vqalstm.BOW(lstm_config)
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  local modules = nn.Parallel()
  modules:add(self.lstm)
  modules:add(self.vqa_module)

  if self.cuda then
    modules = modules:cuda()
  end

  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  if self.structure == 'bilstm' or self.structure == 'bigru' then
    share_params(self.lstm_b, self.lstm)
  end
end

function LSTMVQA:new_vqa_module()
  local input_dim = self.num_layers * self.mem_dim
  local inputs, vec
  if self.structure == 'lstm' or self.structure == 'rlstm' 
    or self.structure == 'gru' or self.structure == 'rnn'
    or self.structure == 'rnnsu' or self.structure == 'bow' then
    local rep = nn.Identity()()
    if self.num_layers == 1 then
      vec = {rep}
    else
      vec = nn.JoinTable(1)(rep)
    end
    inputs = {rep}
  elseif self.structure == 'bilstm' or self.structure == 'bigru' then
    local frep, brep = nn.Identity()(), nn.Identity()()
    input_dim = input_dim * 2
    if self.num_layers == 1 then
      vec = nn.JoinTable(1){frep, brep}
    else
      vec = nn.JoinTable(1){nn.JoinTable(1)(frep), nn.JoinTable(1)(brep)}
    end
    inputs = {frep, brep}
  end

  local logprobs
  if self.dropout then
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(
        nn.Dropout()(vec)))
  else
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(vec))
  end

  return nn.gModule(inputs, {logprobs})
end

function LSTMVQA:train(dataset)
  self.lstm:training()
  self.vqa_module:training()
  if self.structure == 'bilstm' or self.structure == 'bigru' then
    self.lstm_b:training()
  end

  local indices = torch.randperm(dataset.size)
  --local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      if not self.imageonly then
        self.emb:zeroGradParameters()
      end

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local ques = dataset.questions[idx] -- question word indicies
        local ans = dataset.answers[idx]

        -------------------- FORWARD --------------------
        local inputs
        if not self.imageonly then -- textonly or imagetext
          inputs = self.emb:forward(ques) -- question word vectors
        else -- imageonly
          local img = dataset.images[idx]
          local imgfea = dataset.imagefeas[img]
          if imgfea:size():size() == 1 then
            inputs = torch.repeatTensor(imgfea,1,1)
          else
            inputs = imgfea
          end
        end
        if self.imagetext then
          local img = dataset.images[idx]
          local imgfea = dataset.imagefeas[img]
          inputs = self.jointable2:forward{inputs, torch.repeatTensor(imgfea,inputs:size(1),1)}
        end

        -- get representations
        local rep -- htables
        if self.structure == 'lstm' or self.structure == 'gru' or self.structure == 'rnn'
          or self.structure == 'rnnsu' or self.structure == 'bow' then
          rep = self.lstm:forward(inputs)
        elseif self.structure == 'rlstm' then
          rep = self.lstm:forward(inputs, true)
        elseif self.structure == 'bilstm' or self.structure == 'bigru' then
          rep = {
            self.lstm:forward(inputs),
            self.lstm_b:forward(inputs, true), -- true => reverse
          }
        end

        -- compute class log probabilities
        local output = self.vqa_module:forward(rep) -- class log prob

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ans)
        loss = loss + example_loss

        -------------------- BACKWARD --------------------
        local obj_grad = self.criterion:backward(output, ans)
        local rep_grad = self.vqa_module:backward(rep, obj_grad)
        local input_grads
        if self.structure == 'lstm' or self.structure == 'gru' or self.structure == 'rnn' or self.structure == 'rnnsu' or self.structure == 'bow' then
          input_grads = self:LSTM_backward(inputs, rep_grad)
        elseif self.structure == 'rlstm' then
          input_grads = self:rLSTM_backward(inputs, rep_grad, true)
        elseif self.structure == 'bilstm' or self.structure == 'bigru' then
          input_grads = self:BiLSTM_backward(inputs, rep_grad)
        end
        if self.textonly then
          self.emb:backward(ques, input_grads)
        elseif self.imagetext then
          local firstInput = input_grads:narrow(2,1,self.emb_dim):clone()
          self.emb:backward(ques, firstInput)
          local secondInput = input_grads:narrow(2,self.emb_dim+1,self.im_fea_dim):clone()
        end
      end

      -- comment these, since batch_size is 1
      --loss = loss / batch_size
      --self.grad_params:div(batch_size)
      --self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
    if not self.imageonly then
      self.emb:updateParameters(self.emb_learning_rate)
    end
  end
  xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function LSTMVQA:LSTM_backward(inputs, rep_grad)
  local grad
  local numelem = inputs:size(1)
  if self.num_layers == 1 then
    grad = torch.zeros(numelem, self.mem_dim)
    if self.cuda then
      grad = grad:float():cuda()
    end
    grad[numelem] = rep_grad
  else
    grad = torch.zeros(numelem, self.num_layers, self.mem_dim)
    if self.cuda then
      grad = grad:float():cuda()
    end
    for l = 1, self.num_layers do
      grad[{numelem, l, {}}] = rep_grad[l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  return input_grads
end

-- LSTM backward propagation
function LSTMVQA:rLSTM_backward(inputs, rep_grad)
  local grad
  local numelem = inputs:size(1)
  if self.num_layers == 1 then
    grad = torch.zeros(numelem, self.mem_dim)
    if self.cuda then
      grad = grad:float():cuda()
    end
    grad[1] = rep_grad
  else
    grad = torch.zeros(numelem, self.num_layers, self.mem_dim)
    if self.cuda then
      grad = grad:float():cuda()
    end
    for l = 1, self.num_layers do
      grad[{1, l, {}}] = rep_grad[l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad, true)
  return input_grads
end

-- Bidirectional LSTM backward propagation
function LSTMVQA:BiLSTM_backward(inputs, rep_grad)
  local grad, grad_b
  local numelem = inputs:size(1)
  if self.num_layers == 1 then
    grad   = torch.zeros(numelem, self.mem_dim)
    grad_b = torch.zeros(numelem, self.mem_dim)
    if self.cuda then
      grad = grad:float():cuda()
      grad_b = grad_b:float():cuda()
    end
    grad[numelem] = rep_grad[1]
    grad_b[1] = rep_grad[2]
  else
    grad   = torch.zeros(numelem, self.num_layers, self.mem_dim)
    grad_b = torch.zeros(numelem, self.num_layers, self.mem_dim)
    if self.cuda then
      grad = grad:float():cuda()
      grad_b = grad_b:float():cuda()
    end
    for l = 1, self.num_layers do
      grad[{numelem, l, {}}] = rep_grad[1][l]
      grad_b[{1, l, {}}] = rep_grad[2][l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  local input_grads_b = self.lstm_b:backward(inputs, grad_b, true)
  return input_grads + input_grads_b
end

-- Predict the vqa of a sentence.
function LSTMVQA:predict(ques, imgfea)
  self.lstm:evaluate()
  self.vqa_module:evaluate()
  local inputs
  if not self.imageonly then -- textonly or imagetext
    inputs = self.emb:forward(ques) -- question word vectors
  else -- imageonly
    if imgfea:size():size() == 1 then
      inputs = torch.repeatTensor(imgfea,1,1)
    else
      inputs = imgfea
    end
  end
  if self.imagetext then
    inputs = self.jointable2:forward{inputs, torch.repeatTensor(imgfea,inputs:size(1),1)}
  end

  local rep
  if self.structure == 'lstm' or self.structure == 'gru' or self.structure == 'rnn' or self.structure == 'rnnsu' or self.structure == 'bow' then
    rep = self.lstm:forward(inputs)
  elseif self.structure == 'rlstm' then
    rep = self.lstm:forward(inputs, true)
  elseif self.structure == 'bilstm' or self.structure == 'bigru' then
    self.lstm_b:evaluate()
    rep = {
      self.lstm:forward(inputs),
      self.lstm_b:forward(inputs, true),
    }
  end
  local logprobs = self.vqa_module:forward(rep)
  local prediction = argmax(logprobs)
  self.lstm:forget()
  if self.structure == 'bilstm' or self.structure == 'bigru' then
    self.lstm_b:forget()
  end
  return prediction
end

-- Produce vqa predictions for each sentence in the dataset.
function LSTMVQA:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  if self.textonly then
    for i = 1, dataset.size do
      xlua.progress(i, dataset.size)
      predictions[i] = self:predict(dataset.questions[i])
    end
  else
    for i = 1, dataset.size do
      xlua.progress(i, dataset.size)
      predictions[i] = self:predict(dataset.questions[i], dataset.imagefeas[dataset.images[i]])
    end
  end
  return predictions
end

function LSTMVQA:print_config()
  local num_params = self.params:size(1)
  local num_vqa_params = self:new_vqa_module():getParameters():size(1)
  print(string.format('%-25s = %d',   'num params', num_params))
  print(string.format('%-25s = %d',   'num compositional params', num_params - num_vqa_params))
  print(string.format('%-25s = %d',   'LSTM memory dim', self.mem_dim))
  print(string.format('%-25s = %s',   'LSTM structure', self.structure))
  print(string.format('%-25s = %d',   'LSTM layers', self.num_layers))
  print(string.format('%-25s = %.2e', 'regularization strength', self.reg))
  print(string.format('%-25s = %d',   'minibatch size', self.batch_size))
  print(string.format('%-25s = %.2e', 'learning rate', self.learning_rate))
  print(string.format('%-25s = %.2e', 'word vector learning rate', self.emb_learning_rate))
  print(string.format('%-25s = %s',   'dropout', tostring(self.dropout)))
  print(string.format('%-25s = %s',   'cuda', tostring(self.cuda)))
  if not self.imageonly then
    print(string.format('%-25s = %d',   'word vector dim', self.emb_dim))
  end
  if not self.textonly then
    print(string.format('%-25s = %d',   'image feature dim', self.im_fea_dim))
  end
end

--
-- Serialization
--

function LSTMVQA:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    cuda              = self.cuda,
    emb_learning_rate = self.emb_learning_rate,
    learning_rate     = self.learning_rate,
    num_layers        = self.num_layers,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
    im_fea_dim        = self.im_fea_dim,
    num_classes       = self.num_classes,
    textonly          = self.textonly,
    imageonly         = self.imageonly
  }
  if not self.imageonly then
    config.emb_vecs   = self.emb.weight:float()
  end

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function LSTMVQA.load(path)
  local state = torch.load(path)
  --state.config.num_classes = 969--trick
  --state.config.textonly = string.find(path, 'textonly') and true or false--trick
  local model = vqalstm.LSTMVQA.new(state.config)
  model.params:copy(state.params)
  return model
end
