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

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  -- vqa classification module
  self.vqa_module1 = self:new_vqa_module()
  self.vqa_module2 = self:new_vqa_module()
  self.vqa_module3 = self:new_vqa_module()
  self.vqa_module4 = self:new_vqa_module()

  if self.cuda then
    self.criterion = self.criterion:cuda()
    self.vqa_module1 = self.vqa_module1:cuda()
    self.vqa_module2 = self.vqa_module2:cuda()
    self.vqa_module3 = self.vqa_module3:cuda()
    self.vqa_module4 = self.vqa_module4:cuda()
  end

  local modules = nn.Parallel()
  modules:add(self.vqa_module1)
  modules:add(self.vqa_module2)
  modules:add(self.vqa_module3)
  modules:add(self.vqa_module4)

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
  self.vqa_module1:training()
  self.vqa_module2:training()
  self.vqa_module3:training()
  self.vqa_module4:training()
  
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
        local output
        if typ == 0 then
          output = self.vqa_module1:forward(imgfea) -- class log prob
        elseif typ == 1 then
          output = self.vqa_module2:forward(imgfea) -- class log prob
        elseif typ == 2 then
          output = self.vqa_module3:forward(imgfea) -- class log prob
        elseif typ == 3 then
          output = self.vqa_module4:forward(imgfea) -- class log prob
        else
          error("Error type in prediction: "..typ)
        end

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ans)
        loss = loss + example_loss

        -------------------- BACKWARD --------------------
        local obj_grad = self.criterion:backward(output, ans)
        local rep_grad
        if typ == 0 then
          rep_grad = self.vqa_module1:backward(imgfea, obj_grad)
        elseif typ == 1 then
          rep_grad = self.vqa_module2:backward(imgfea, obj_grad)
        elseif typ == 2 then
          rep_grad = self.vqa_module3:backward(imgfea, obj_grad)
        elseif typ == 3 then
          rep_grad = self.vqa_module4:backward(imgfea, obj_grad)
        else
          error("Error type in prediction: "..typ)
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
  end
  xlua.progress(dataset.size, dataset.size)
end

-- Predict the vqa of a sentence.
function ImageVQA:predict(typ, imgfea)
  local logprobs
  if typ == 0 then
    self.vqa_module1:evaluate()
    logprobs = self.vqa_module1:forward(imgfea)
  elseif typ == 1 then
    self.vqa_module2:evaluate()
    logprobs = self.vqa_module2:forward(imgfea)
  elseif typ == 2 then
    self.vqa_module3:evaluate()
    logprobs = self.vqa_module3:forward(imgfea)
  elseif typ == 3 then
    self.vqa_module4:evaluate()
    logprobs = self.vqa_module4:forward(imgfea)
  else
    error("Error type in prediction: "..typ)
  end
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
