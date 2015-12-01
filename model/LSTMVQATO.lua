--[[

  VQA using LSTMs.

--]]

local LSTMVQATO, parent = torch.class('vqalstm.LSTMVQATO', 'vqalstm.LSTMVQA')

function LSTMVQATO:__init(config)
  self.mem_dim           = config.mem_dim           or 150
  self.learning_rate     = config.learning_rate     or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.num_layers        = config.num_layers        or 1
  self.batch_size        = config.batch_size        or 1 -- train per 1 sample
  self.reg               = config.reg               or 1e-4
  self.structure         = config.structure         or 'lstm' -- {lstm, bilstm}
  self.dropout           = (config.dropout == nil) and true or config.dropout
  self.num_classes       = config.num_classes
  self.cuda              = config.cuda              or false
  assert(self.num_classes~=nil)

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.in_zeros = torch.zeros(self.emb_dim)

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  -- vqa classification module
  self.vqa_module = self:new_vqa_module()

  if self.cuda then
    self.emb = self.emb:cuda()
    self.in_zeros = self.in_zeros:float():cuda()
    self.criterion = self.criterion:cuda()
    self.vqa_module = self.vqa_module:cuda()
  end

  -- initialize LSTM model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = true,
    cuda = self.cuda
  }

  if self.structure == 'lstm' then
    self.lstm = vqalstm.LSTM(lstm_config)
  elseif self.structure == 'bilstm' then
    self.lstm = vqalstm.LSTM(lstm_config)
    self.lstm_b = vqalstm.LSTM(lstm_config)
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  local modules = nn.Parallel()
    :add(self.lstm)
    :add(self.vqa_module)

  if self.cuda then
    modules = modules:cuda()
  end

  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  if self.structure == 'bilstm' then
    share_params(self.lstm_b, self.lstm)
  end
end

function LSTMVQATO:new_vqa_module()
  return parent.new_vqa_module(self)
end

function LSTMVQATO:train(dataset)
  self.lstm:training()
  self.vqa_module:training()
  if self.structure == 'bilstm' then
    self.lstm_b:training()
  end

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
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

        -------------------- FORWARD --------------------
        local inputs = self.emb:forward(ques) -- question word vectors

        -- get sentence representations
        local rep -- htables
        if self.structure == 'lstm' then
          rep = self.lstm:forward(inputs)
        elseif self.structure == 'bilstm' then
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
        if self.structure == 'lstm' then
          input_grads = self:LSTM_backward(ques, inputs, rep_grad)
        elseif self.structure == 'bilstm' then
          input_grads = self:BiLSTM_backward(ques, inputs, rep_grad)
        end
        self.emb:backward(ques, input_grads)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

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
function LSTMVQATO:LSTM_backward(ques, inputs, rep_grad)
  return parent.LSTM_backward(self, ques, inputs, rep_grad)
end

-- Bidirectional LSTM backward propagation
function LSTMVQATO:BiLSTM_backward(ques, inputs, rep_grad)
  return parent.BiLSTM_backward(self, ques, inputs, rep_grad)
end

-- Predict the vqa of a sentence.
function LSTMVQATO:predict(ques)
  self.lstm:evaluate()
  self.vqa_module:evaluate()
  local inputs = self.emb:forward(ques)

  local rep
  if self.structure == 'lstm' then
    rep = self.lstm:forward(inputs)
  elseif self.structure == 'bilstm' then
    self.lstm_b:evaluate()
    rep = {
      self.lstm:forward(inputs),
      self.lstm_b:forward(inputs, true),
    }
  end
  local logprobs = self.vqa_module:forward(rep)
  local prediction = argmax(logprobs)
  self.lstm:forget()
  if self.structure == 'bilstm' then
    self.lstm_b:forget()
  end
  return prediction
end

-- Produce vqa predictions for each sentence in the dataset.
function LSTMVQATO:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.questions[i])
  end
  return predictions
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function LSTMVQATO:print_config()
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
end

--
-- Serialization
--

function LSTMVQATO:save(path)
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
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function LSTMVQATO.load(path)
  local state = torch.load(path)
  local model = vqalstm.LSTMVQATO.new(state.config)
  model.params:copy(state.params)
  return model
end
