require 'gnuplot'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Script for plot accuracy v.s. epoch.')
cmd:text()
cmd:text('Options')
cmd:option('-f','train/train_vqalstm.lua-2015-12-01T165953.log','Log file name')
cmd:option('-p',false,'Plot figure')
cmd:option('-g',true,'Grid')
cmd:text()

local args = cmd:parse(arg)
--local file_path = (args.f==nil) and 'train/train_vqalstm.lua-2015-12-01T165953.log' or args.f
local file_path = args.f
local grid = (args.g==nil) or args.g
--file_path = 'train/train_vqalstm.lua-2015-12-01T165953.log'
--file_path = 'train/train_vqalstm_textonly.lua-2015-12-01T091138.log'

-- read log file
local f = assert(io.open(file_path, 'r'))
local file_content = f:read('*all')
f:close()

-- extract values
local epoch, tr_score, tt_score = {}, {}, {}
for ep,tr,tt in string.gmatch(file_content,'-- epoch (%d+).-\n-- train score: (%d*%.?%d+).-\n-- test score: (%d?%.?%d+)') do
  --print(ep,tr,tt)
  table.insert(epoch, ep)
  table.insert(tr_score, tr)
  table.insert(tt_score, tt)
end

epoch = torch.Tensor(epoch)
tr_score = torch.Tensor(tr_score)
tt_score = torch.Tensor(tt_score)

-- plot
if args.p then
  local svg_file = file_path ..'.svg'
  print('Save figure to: '.. svg_file)
  gnuplot.svgfigure(svg_file)
  gnuplot.plot({'Train Accuracy',epoch,tr_score,'~'}, {'Test Accuracy',epoch,tt_score,'~'})
  gnuplot.xlabel('epoch')
  gnuplot.ylabel('accuracy')
  gnuplot.grid(grid)
  gnuplot.title(file_path)
  gnuplot.plotflush()
end

-- max test accuracy
local maxepoch = 1
local maxtracc = 0
local maxttacc = 0
for i=1,epoch:size(1) do
  if tt_score[i] > maxttacc then
  	maxepoch = epoch[i]
  	maxtracc = tr_score[i]
  	maxttacc = tt_score[i]
  end
end
print('Best test accuracy:')
print('at epoch '.. maxepoch)
print('-- train score: '.. maxtracc)
print('-- test score: '.. maxttacc)
