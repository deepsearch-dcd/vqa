require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('cunn')
--require('sys')
--require('lfs')
COCOQA = require 'dataset/COCOQA'
DAQUAR = require 'dataset/DAQUAR'
npy4th = require 'npy4th'
require 'util/Set'
require 'util/DataLoad'

vqalstm = {}

include('module/fLSTM.lua')
include('module/fGRU.lua')
include('module/fRNN.lua')
include('module/fRNNSU.lua')
include('module/fBOW.lua')
include('model/LSTMVQA.lua')
include('model/ConcatVQA.lua')
include('model/ImageVQA.lua')

-- global paths (modify if desired)
--vqalstm.data_dir        = 'data'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
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

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

header('init function being called ...')

-- some useful functions
function accuracy(pred, gold) -- both are torch.Tensor
  return torch.eq(pred, gold):sum() / pred:size(1)
end
