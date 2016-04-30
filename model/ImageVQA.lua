--[[

  VQA using LSTMs with image only.

--]]

local ImageVQA = torch.class('vqalstm.ImageVQA')

function ImageVQA:__init(config)
  self.learning_rate     = config.learning_rate     or 0.05
  self.batch_size        = config.batch_size        or 1 -- train per 1 sample
  self.reg               = config.reg               or 1e-4
  self.dropout           = (config.dropout == nil) and true or config.dropout
  self.num_classes       = config.num_classes
  self.cuda              = config.cuda              or false
  self.im_fea_dim        = config.im_fea_dim        or 1000
  assert(self.num_classes~=nil)
  self.num_model         = config.num_model         or 4

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  -- vqa classification module
  self.vqa_module = {}
  for i = 1,self.num_model do
    self.vqa_module[i] = self:new_vqa_module()
  end

  if self.cuda then
    self.criterion = self.criterion:cuda()
    for i = 1,self.num_model do
      self.vqa_module[i] = self.vqa_module[i]:cuda()
    end
  end

  local modules = nn.Parallel()
  for i = 1,self.num_model do
    modules:add(self.vqa_module[i])
  end

  if self.cuda then
    modules = modules:cuda()
  end

  self.params, self.grad_params = modules:getParameters()
end

function ImageVQA:new_vqa_module()
  local input_dim = self.im_fea_dim
  local inputs, vec
  local rep = nn.Identity()()
  vec = {rep}
  inputs = {rep}

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

function ImageVQA:train(dataset)
  for i = 1,self.num_model do
    self.vqa_module[i]:training()
  end
  
  local indices = torch.randperm(dataset.size)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local ans = dataset.answers[idx]
        local typ = tonumber(dataset.types[idx])

        -------------------- FORWARD --------------------
        local img = dataset.images[idx]
        local imgfea = dataset.imagefeas[img]

        -- compute class log probabilities
        local output = self.vqa_module[typ+1]:forward(imgfea) -- class log prob

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ans)
        loss = loss + example_loss

        -------------------- BACKWARD --------------------
        local obj_grad = self.criterion:backward(output, ans)
        local rep_grad = self.vqa_module[typ+1]:backward(imgfea, obj_grad)
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
  end
  xlua.progress(dataset.size, dataset.size)
end

-- Predict the vqa of a sentence.
function ImageVQA:predict(typ, imgfea)
  self.vqa_module[typ+1]:evaluate()
  local logprobs = self.vqa_module[typ+1]:forward(imgfea)
  local prediction = argmax(logprobs)
  return prediction
end

-- Produce vqa predictions for each sentence in the dataset.
function ImageVQA:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(tonumber(dataset.types[i]), dataset.imagefeas[dataset.images[i]])
  end
  return predictions
end

function ImageVQA:print_config()
  local num_params = self.params:size(1)
  local num_vqa_params = self:new_vqa_module():getParameters():size(1)
  print(string.format('%-25s = %d',   'num params', num_params))
  print(string.format('%-25s = %d',   'num compositional params', num_params - num_vqa_params))
  print(string.format('%-25s = %.2e', 'regularization strength', self.reg))
  print(string.format('%-25s = %d',   'minibatch size', self.batch_size))
  print(string.format('%-25s = %.2e', 'learning rate', self.learning_rate))
  print(string.format('%-25s = %s',   'dropout', tostring(self.dropout)))
  print(string.format('%-25s = %s',   'cuda', tostring(self.cuda)))
  print(string.format('%-25s = %s',   'image (only) feature dim', self.im_fea_dim))
end

--
-- Serialization
--

function ImageVQA:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    cuda              = self.cuda,
    learning_rate     = self.learning_rate,
    reg               = self.reg,
    im_fea_dim        = self.im_fea_dim,
    num_classes       = self.num_classes,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function ImageVQA.load(path)
  local state = torch.load(path)
  --state.config.num_classes = 969--trick
  local model = vqalstm.ImageVQA.new(state.config)
  model.params:copy(state.params)
  return model
end
