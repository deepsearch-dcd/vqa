--[[

  VQA using LSTMs.

--]]

local ConcatVQA = torch.class('vqalstm.ConcatVQA')

function ConcatVQA:__init(config)
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
  self.im_fea_dim        = config.im_fea_dim        or 1000
  assert(self.num_classes~=nil)

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  -- vqa classification module
  self.vqa_module = self:new_vqa_module()

  self.jointable1 = nn.JoinTable(1)
  self.narrow1 = nn.Narrow(1,1,self.mem_dim)
  self.narrow1_im = nn.Narrow(1,self.mem_dim+1,self.mem_dim)

  if self.cuda then
    self.emb = self.emb:cuda()
    self.criterion = self.criterion:cuda()
    self.vqa_module = self.vqa_module:cuda()
    self.jointable1 = self.jointable1:cuda()
    self.narrow1 = self.narrow1:cuda()
    self.narrow1_im = self.narrow1_im:cuda()
  end

  -- initialize LSTM model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = true,
    cuda = self.cuda
  }
  local lstm_config_im = {
    in_dim = self.im_fea_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = true,
    cuda = self.cuda
  }

  if self.structure == 'lstm' or self.structure == 'rlstm' then
    self.lstm = vqalstm.LSTM(lstm_config)
    self.lstm_im = vqalstm.LSTM(lstm_config_im)
  elseif self.structure == 'bilstm' then
    self.lstm = vqalstm.LSTM(lstm_config)
    self.lstm_b = vqalstm.LSTM(lstm_config)
    self.lstm_im = vqalstm.LSTM(lstm_config_im)
    self.lstm_im_b = vqalstm.LSTM(lstm_config_im)
  elseif self.structure == 'gru' then
    self.lstm = vqalstm.GRU(lstm_config)
    self.lstm_im = vqalstm.GRU(lstm_config_im)
  elseif self.structure == 'bigru' then
    self.lstm = vqalstm.GRU(lstm_config)
    self.lstm_b = vqalstm.GRU(lstm_config)
    self.lstm_im = vqalstm.GRU(lstm_config_im)
    self.lstm_im_b = vqalstm.GRU(lstm_config_im)
  elseif self.structure == 'rnn' then
    self.lstm = vqalstm.RNN(lstm_config)
    self.lstm_im = vqalstm.RNN(lstm_config_im)
  elseif self.structure == 'rnnsu' then
    self.lstm = vqalstm.RNNSU(lstm_config)
    self.lstm_im = vqalstm.RNNSU(lstm_config_im)
  elseif self.structure == 'bow' then
    self.lstm = vqalstm.BOW(lstm_config)
    self.lstm_im = vqalstm.BOW(lstm_config_im)
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  local modules = nn.Parallel()
  modules:add(self.lstm)
  modules:add(self.lstm_im)
  modules:add(self.vqa_module)

  if self.cuda then
    modules = modules:cuda()
  end

  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  if self.structure == 'bilstm' or self.structure == 'bigru' then
    share_params(self.lstm_b, self.lstm)
    share_params(self.lstm_im_b, self.lstm_im)
  end
end

function ConcatVQA:new_vqa_module()
  local input_dim
  input_dim = self.num_layers * (self.mem_dim * 2)
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

function ConcatVQA:train(dataset)
  self.lstm:training()
  self.lstm_im:training()
  self.vqa_module:training()
  if self.structure == 'bilstm' or self.structure == 'bigru' then
    self.lstm_b:training()
    self.lstm_im_b:training()
  end

  local indices = torch.randperm(dataset.size)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local ques = dataset.questions[idx] -- question word indicies
        local ans = dataset.answers[idx]
        -- image features
        local img = dataset.images[idx]
        local imgfea = dataset.imagefeas[img]
        if imgfea:size():size() == 1 then
          imgfea = torch.repeatTensor(imgfea,1,1)
        end

        -------------------- FORWARD --------------------
        local inputs = self.emb:forward(ques) -- question word vectors

        -- get sentence representations
        local rep -- htables
        local rep_im
        if self.structure == 'lstm' or self.structure == 'gru' or self.structure == 'rnn'
          or self.structure == 'rnnsu' or self.structure == 'bow' then
          rep = self.lstm:forward(inputs)
          rep_im = self.lstm_im:forward(imgfea)
        elseif self.structure == 'rlstm' then
          rep = self.lstm:forward(inputs, true)
          rep_im = self.lstm_im:forward(imgfea, true)
        elseif self.structure == 'bilstm' or self.structure == 'bigru' then
          rep = {
            self.lstm:forward(inputs),
            self.lstm_b:forward(inputs, true), -- true => reverse
          }
          rep_im = {
            self.lstm_im:forward(imgfea),
            self.lstm_im_b:forward(imgfea, true), -- true => reverse
          }
        end

        if self.structure == 'bilstm' or self.structure == 'bigru' then
          if self.num_layers == 1 then
            rep[1] = self.jointable1:forward{rep[1], rep_im[1]}
            rep[2] = self.jointable1:forward{rep[2], rep_im[2]}
          else -- num_layers > 1
            for i = 1,self.num_layers do
              rep[1][i] = self.jointable1:forward{rep[1][i], rep_im[1][i]}
              rep[2][i] = self.jointable1:forward{rep[2][i], rep_im[2][i]}
            end
          end
        else -- structure is not bilstm
          if self.num_layers == 1 then
            rep = self.jointable1:forward{rep, rep_im}
          else -- num_layers > 1
            for i = 1,self.num_layers do
              rep[i] = self.jointable1:forward{rep[i], rep_im[i]}
            end
          end
        end

        -- compute class log probabilities
        local output = self.vqa_module:forward(rep) -- class log prob

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ans)
        loss = loss + example_loss

        -------------------- BACKWARD --------------------
        local obj_grad = self.criterion:backward(output, ans)
        local rep_grad = self.vqa_module:backward(rep, obj_grad)
        local rep_im_grad = {}

        if self.structure == 'bilstm' or self.structure == 'bigru' then
          if self.num_layers == 1 then
            rep_im_grad[1] = self.narrow1_im:forward(rep_grad[1])
            rep_im_grad[2] = self.narrow1_im:forward(rep_grad[2])
            rep_grad[1] = self.narrow1:forward(rep_grad[1])
            rep_grad[2] = self.narrow1:forward(rep_grad[2])
          else -- num_layers > 1
            rep_im_grad[1] = {}
            rep_im_grad[2] = {}
            for i = 1,self.num_layers do
              rep_im_grad[1][i] = self.narrow1_im:forward(rep_grad[1][i])
              rep_im_grad[2][i] = self.narrow1_im:forward(rep_grad[2][i])
              rep_grad[1][i] = self.narrow1:forward(rep_grad[1][i])
              rep_grad[2][i] = self.narrow1:forward(rep_grad[2][i])
            end
          end
        else -- structure is not bilstm
          if self.num_layers == 1 then
            rep_im_grad = self.narrow1_im:forward(rep_grad)
            rep_grad = self.narrow1:forward(rep_grad)
          else -- num_layers > 1
            for i = 1,self.num_layers do
              rep_im_grad[i] = self.narrow1_im:forward(rep_grad[i])
              rep_grad[i] = self.narrow1:forward(rep_grad[i])
            end
          end
        end

        local input_grads
        if self.structure == 'lstm' or self.structure == 'gru' or self.structure == 'rnn' or self.structure == 'rnnsu' or self.structure == 'bow' then
          input_grads = self:LSTM_backward(inputs, rep_grad)
          self:LSTM_backward(imgfea, rep_im_grad, true)
        elseif self.structure == 'rlstm' then
          input_grads = self:rLSTM_backward(inputs, rep_grad)
          self:rLSTM_backward(imgfea, rep_im_grad, true)
        elseif self.structure == 'bilstm' or self.structure == 'bigru' then
          input_grads = self:BiLSTM_backward(inputs, rep_grad)
          self:BiLSTM_backward(imgfea, rep_im_grad, true)
        end
        self.emb:backward(ques, input_grads)
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
    self.emb:updateParameters(self.emb_learning_rate)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function ConcatVQA:LSTM_backward(inputs, rep_grad, is_im)
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
  local input_grads
  if not is_im then
    input_grads = self.lstm:backward(inputs, grad)
  else
    input_grads = self.lstm_im:backward(inputs, grad)
  end
  return input_grads
end

-- LSTM backward propagation
function ConcatVQA:rLSTM_backward(inputs, rep_grad, is_im)
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
  local input_grads
  if not is_im then
    input_grads = self.lstm:backward(inputs, grad, true)
  else
    input_grads = self.lstm_im:backward(inputs, grad, true)
  end
  return input_grads
end

-- Bidirectional LSTM backward propagation
function ConcatVQA:BiLSTM_backward(inputs, rep_grad, is_im)
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
  local input_grads
  local input_grads_b
  if not is_im then
    input_grads = self.lstm:backward(inputs, grad)
    input_grads_b = self.lstm_b:backward(inputs, grad_b, true)
  else
    input_grads = self.lstm_im:backward(inputs, grad)
    input_grads_b = self.lstm_im_b:backward(inputs, grad_b, true)
  end
  return input_grads + input_grads_b
end

-- Predict the vqa of a sentence.
function ConcatVQA:predict(ques, imgfea)
  self.lstm:evaluate()
  self.lstm_im:evaluate()
  self.vqa_module:evaluate()

  local inputs = self.emb:forward(ques)
  if imgfea:size():size() == 1 then
    imgfea = torch.repeatTensor(imgfea,1,1)
  end

  local rep
  if self.structure == 'lstm' or self.structure == 'gru' or self.structure == 'rnn' or self.structure == 'rnnsu' or self.structure == 'bow' then
    rep = self.lstm:forward(inputs)
    rep_im = self.lstm_im:forward(imgfea)
  elseif self.structure == 'rlstm' then
    rep = self.lstm:forward(inputs, true)
    rep_im = self.lstm_im:forward(imgfea, true)
  elseif self.structure == 'bilstm' or self.structure == 'bigru' then
    self.lstm_b:evaluate()
    self.lstm_im_b:evaluate()
    rep = {
      self.lstm:forward(inputs),
      self.lstm_b:forward(inputs, true),
    }
    rep_im = {
      self.lstm_im:forward(imgfea),
      self.lstm_im_b:forward(imgfea, true), -- true => reverse
    }
  end

  if self.structure == 'bilstm' or self.structure == 'bigru' then
    if self.num_layers == 1 then
      for i = 1,self.num_layers do
        rep[1] = self.jointable1:forward{rep[1], rep_im[1]}
        rep[2] = self.jointable1:forward{rep[2], rep_im[2]}
      end
    else -- num_layers > 1
      for i = 1,self.num_layers do
        rep[1][i] = self.jointable1:forward{rep[1][i], rep_im[1][i]}
        rep[2][i] = self.jointable1:forward{rep[2][i], rep_im[2][i]}
      end
    end
  else -- structure is not bilstm
    if self.num_layers == 1 then
      for i = 1,self.num_layers do
        rep = self.jointable1:forward{rep, rep_im}
      end
    else -- num_layers > 1
      for i = 1,self.num_layers do
        rep[i] = self.jointable1:forward{rep[i], rep_im[i]}
      end
    end
  end

  local logprobs = self.vqa_module:forward(rep)
  local prediction = argmax(logprobs)
  self.lstm:forget()
  self.lstm_im:forget()
  if self.structure == 'bilstm' or self.structure == 'bigru' then
    self.lstm_b:forget()
    self.lstm_im_b:forget()
  end
  return prediction
end

-- Produce vqa predictions for each sentence in the dataset.
function ConcatVQA:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.questions[i], dataset.imagefeas[dataset.images[i]])
  end
  return predictions
end

function ConcatVQA:print_config()
  local num_params = self.params:size(1)
  local num_vqa_params = self:new_vqa_module():getParameters():size(1)
  print(string.format('%-25s = %d',   'num params', num_params))
  print(string.format('%-25s = %d',   'num compositional params', num_params - num_vqa_params))
  print(string.format('%-25s = %d',   'word vector dim', self.emb_dim))
  print(string.format('%-25s = %d',   'LSTM memory dim', self.mem_dim))
  print(string.format('%-25s = %s',   'LSTM structure', self.structure))
  print(string.format('%-25s = %d',   'LSTM layers', self.num_layers))
  print(string.format('%-25s = %.2e', 'regularization strength', self.reg))
  print(string.format('%-25s = %d',   'minibatch size', self.batch_size))
  print(string.format('%-25s = %.2e', 'learning rate', self.learning_rate))
  print(string.format('%-25s = %.2e', 'word vector learning rate', self.emb_learning_rate))
  print(string.format('%-25s = %s',   'dropout', tostring(self.dropout)))
  print(string.format('%-25s = %s',   'cuda', tostring(self.cuda)))
  print(string.format('%-25s = %s',   'image feature dim', self.im_fea_dim))
end

--
-- Serialization
--

function ConcatVQA:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    cuda              = self.cuda,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    learning_rate     = self.learning_rate,
    num_layers        = self.num_layers,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
    im_fea_dim        = self.im_fea_dim,
    num_classes       = self.num_classes,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function ConcatVQA.load(path)
  local state = torch.load(path)
  --state.config.num_classes = 969--trick
  local model = vqalstm.ConcatVQA.new(state.config)
  model.params:copy(state.params)
  return model
end
