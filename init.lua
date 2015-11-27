require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('cunn')
--require('sys')
--require('lfs')
DAQUAR = require 'dataset/DAQUAR'
npy4th = require 'npy4th'

vqalstm = {}

--include('util/read_data.lua')
--include('util/Tree.lua')
--include('util/Vocab.lua')
--include('layers/CRowAddTable.lua')
include('module/fLSTM.lua')
include('model/LSTMVQA.lua')
--include('model/ChildSumTreeLSTM.lua')
--include('model/BinaryTreeLSTM.lua')
--include('relatedness/LSTMSim.lua')
--include('relatedness/TreeLSTMSim.lua')
--include('sentiment/LSTMSentiment.lua')
--include('sentiment/TreeLSTMSentiment.lua')

printf = utils.printf
--printf=function(...) print(string.format(...)) end

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
        --print('module: '..torch.type(node.data.module.weight or node.data.module.bias))
        --print('src: '..torch.type(src.forwardnodes[i].data.module.weight or src.forwardnodes[i].data.module.bias))
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    --print('module: '..torch.type(node.data.module.weight or node.data.module.bias))
    --print('src: '..torch.type(src.forwardnodes[i].data.module.weight or src.forwardnodes[i].data.module.bias))
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 75)) -- 80
  print(s)
  print(string.rep('-', 75)) -- 80
end

header('init function being called ...')
