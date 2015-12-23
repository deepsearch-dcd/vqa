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


vqalstm = {}

include('module/fLSTM.lua')
include('module/fRNN.lua')
include('module/fRNNSU.lua')
include('model/LSTMVQA.lua')

--printf = utils.printf

-- global paths (modify if desired)
--vqalstm.data_dir        = 'data'
--vqalstm.models_dir      = 'trained_models'
--vqalstm.predictions_dir = 'predictions'

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
