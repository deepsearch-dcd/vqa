require 'optim'
require 'paths'

require 'util/Plotter'
local util = require 'util/util'

-- generate a log file name
-- this function can just run in linux.
local function gen_log_name()
    -- get hostname
    local f = io.popen('/bin/hostname')
    local hostname = f:read() or 'unknown-hostname'
    f:close()

    -- get username
    local username = os.getenv('USER') or 'unkown-user'

    -- get process id
    io.input('/proc/self/stat')
    local pid = io.read('*number')
    io.input(stdin)

    -- get date and time
    local dt = os.date('%Y%m%d-%H%M%S')

    return string.format('%s.%s.%s.%s', hostname, username, dt, pid)
end

-- log evaluation result
local function log(tag, epoch, iter, loss, min_loss, acc, max_acc)
    local LOG_FORMAT = '(%s)Epoch %d, Iteration %d, loss = %.4f(%.4f), ' ..
                       'acc = %.4f(%.4f)'
    print(string.format(LOG_FORMAT, tag, epoch, iter, loss, 
                        min_loss, acc, max_acc))
end

-- test if in the batch mode and get the number of class.
local function get_info_from_output(model, dataset)
    local batch_size, nanswer
    dataset:reset()
    local input, _ = dataset:next()
    local output = model:forward(input)
    if output:dim() == 1 then
        batch_size = 1
        nanswer = output:size(1)
    elseif output:dim() == 2 then
        batch_size = output:size(1)
        nanswer = output:size(2)
    else
        error('The output of the model is expected has the dim == 1 or 2.')
    end
    dataset:reset()
    return batch_size, nanswer
end

-- evaluate model on the given testset
local function test(model, criterion, testset)
    
    local batch_size, nanswer = get_info_from_output(model, testset)
    
    -- initialize confusion matrix
    local confusion = optim.ConfusionMatrix(nanswer)
    local confusion_add = confusion.add
    if batch_size > 1 then
        confusion_add = confusion.batchAdd
    end

    -- initialize total loss
    local total_loss = 0

    -- loop testset
    testset:reset()
    for i = 1, testset:size() do
        -- get data
        local x, t = testset:next()
        local loss = criterion:forward(model:forward(x), t)
        confusion_add(confusion, model.output, t)
        total_loss = total_loss + loss
    end

    confusion:updateValids()
    return total_loss/testset:size(), confusion.totalValid
end

local function get_plotter(save_dir, name, tag)
    local plotter = Plotter(save_dir, name)
    plotter.tag = tag or plotter.tag
    plotter:setNames('train iteration', 'train epoch', 'test epoch')
    return plotter
end

local function get_max(a,b)
    if a > b then return a else return b end
end

local function get_min(a,b)
    if a < b then return a else return b end
end

local function mkdir(path)
    os.execute('mkdir -p "' .. path .. '"')
end

--[[
   train the given model
   
   the given dataset should have the function as follows:
    size()          return the number of samples.
    reset()         reset the index of current sample.
    next()          return the next sample (x, t).
    cuda()          change the tensor in dataset to cudaTensor.

   the keyword of the opt:
    `opt.seed`              option[1234]
    `opt.gpuid`             option[0]
    `opt.home_dir`          option[nil]             if given, prefix to log_dir, plot_dir and cp_dir
    `opt.log_dir`           option[nil]
    `opt.plot_dir`          option['done']
    `opt.cp_dir`            option['done']
    `opt.tag`               option[default]         distinguish different training
    `opt.display_interval`  option[nil]
    `opt.quiet`             option[false]
    `opt.max_epoch`         option[infinite]
    `opt.learningRate`      option[0.001]
    `opt.weightDecay`       option[0.0005]
    `opt.momentum           option[0.9]
    `opt.check_point`       option[nil]
    `opt.pretrained_model`  option[nil]
--]]
function train(opt, model, criterion, trainset, testset)
    -- preprocess the arguments in opt
    opt.learningRate = opt.learningRate or 1e-3
    opt.weightDecay = opt.weightDecay or 5e-4
    opt.momentum = opt.momentum or 0.9
    opt.gpuid = opt.gpuid or -1
    if opt.home_dir then
        local concat = paths.concat
        if opt.plot_dir then
           opt.plot_dir = concat(opt.home_dir, opt.plot_dir) 
        end
        if opt.log_dir then
            opt.log_dir = concat(opt.home_dir, opt.log_dir)
        end
        if opt.cp_dir then
            opt.cp_dir = concat(opt.home_dir, opt.cp_dir)
        end
    end
    opt.plot_dir = opt.plot_dir or 'done'
    opt.tag = opt.tag or 'default'
    opt.cp_dir = opt.cp_dir or 'done'
    mkdir(opt.plot_dir)
    if opt.check_point then
        mkdir(opt.cp_dir)
    end

    -- trigger loging
    if opt.log_dir then
        mkdir(opt.log_dir)
        local cmd = torch.CmdLine()
        local log_name = gen_log_name()
        cmd:log(paths.concat(opt.log_dir, log_name), opt)
        cmd:addTime('vqa', '%F %T')
    end

    -- load pretrained_model
    if opt.pretrained_model then
        print('load pretrained model from ' .. opt.pretrained_model)
        model = torch.load(opt.pretrained_model)
    end
    
    -- initialize cunn and cutorch
    -- modified from char-rnn[https://github.com/karpathy/char-rnn/blob/master/train.lua]
    if opt.gpuid >= 0 then
        local ok, cunn = pcall(require, 'cunn')
        local ok2, cutorch = pcall(require, 'cutorch')
        if not ok then print('package cunn not found!') end
        if not ok2 then print('package cutorch not found!') end
        if ok and ok2 then
            if not opt.quiet then
                print('using CUDA on GPU ' .. opt.gpuid .. '...')
            end
            cutorch.setDevice(opt.gpuid+1) -- beacause lua index from 1
            cutorch.manualSeed(opt.seed or 1234)
        else
            print('Falling back on CPU mode')
            opt.gpuid = -1
        end

        -- change model and dataset to cuda mode.
        model = model:cuda()
        criterion = criterion:cuda()
        trainset:cuda()
        if testset then
            testset:cuda()
        end
    end

    -- get parameters and gradients from model
    local parameters, gradParameters = model:getParameters()
    
    -- configure optim train state
    local config = {}
    config.learningRate = opt.learningRate
    config.weightDecay = opt.weightDecay
    config.momentum = opt.momentum

    -- get the number of class and batch
    local batch_size, nanswer = get_info_from_output(model, trainset)

    -- initialize some variables used to control the training loop
    
    -- the total epoch and iteration util now
    local nepoch, niter = 0, 0
    -- the interval of iterations to display the result of evaluation
    local display_interval = opt.display_interval
    -- accumulated loss for the interval of the iteration and epoch
    local iter_loss, epoch_loss = 0, 0
    -- statistic for evaluation
    local min_train_loss, min_test_loss, 
          max_train_acc, max_test_acc = 1e12, 1e12, 0, 0
    -- iteration per each epoch
    local iters_per_epoch = math.ceil(trainset:size()/batch_size)

    -- evaluation tools
    local iter_confusion = optim.ConfusionMatrix(nanswer)
    local epoch_confusion = optim.ConfusionMatrix(nanswer)
    local confusion_add = iter_confusion.add
    if batch_size > 1 then
        confusion_add = iter_confusion.batchAdd
    end

    -- plot tools
    local acc_plotter = get_plotter(opt.plot_dir, 'acc', opt.tag)
    local loss_plotter = get_plotter(opt.plot_dir, 'loss', opt.tag)

    local function loop()
    -- loop epoch
    while true do
        -- reset trainset
        trainset:reset()

        -- current epoch
        nepoch = nepoch + 1

        -- loop iteration per epoch
        for i = 1, iters_per_epoch do
            -- current iteration
            niter = niter + 1
           
            -- create closure to evaluate f(x) and df/dx
            local feval = function(new_params)
                -- get new parameters
                if new_params ~= parameters then
                    parameters:copy(new_params)
                end
                model:training()
                
                -- get data
                local x, t = trainset:next()

                -- forward
                local loss = criterion:forward(model:forward(x), t)
                
                -- accumulate loss for iteration and epoch respectively
                iter_loss = iter_loss + loss
                epoch_loss = epoch_loss + loss

                -- update confusion
                confusion_add(iter_confusion, model.output, t)
                confusion_add(epoch_confusion, model.output, t)

                --reset gradients
                gradParameters:zero()

                --backward
                model:backward(x, criterion:backward(model.output, t))

                return loss, gradParameters
            end
            
            -- optimize on current mini-batch
            optim.sgd(feval, parameters, config)

            -- display the evaluation of the last interval
            if not opt.quiet 
               and display_interval 
               and niter % display_interval == 0 then

                iter_confusion:updateValids()
                local train_loss = iter_loss / display_interval
                log('train#1', nepoch, niter, train_loss, 0, 
                    iter_confusion.totalValid, 0)
                
                acc_plotter:add{['train iteration'] = 
                    {niter, iter_confusion.totalValid}}
                loss_plotter:add{['train iteration'] = {niter, train_loss}}
                
                iter_loss = 0
                iter_confusion:zero()
            end
        end

        -- display trainset evaluation result
        if not opt.quiet then
            epoch_confusion:updateValids()
            local train_loss = epoch_loss /iters_per_epoch
            min_train_loss = get_min(min_train_loss, train_loss)
            max_train_acc = get_max(max_train_acc, epoch_confusion.totalValid)
            log('train#2', nepoch, niter, train_loss, min_train_loss, 
                epoch_confusion.totalValid, max_train_acc)
            acc_plotter:add{['train epoch'] = {niter, 
                                               epoch_confusion.totalValid}}
            loss_plotter:add{['train epoch'] = {niter, train_loss}}
            epoch_loss = 0
            epoch_confusion:zero()

            if testset then
                model:evaluate()
                -- display testset evaluation result
                local test_loss, test_acc = test(model, criterion, testset)
                min_test_loss = get_min(min_test_loss, test_loss)
                max_test_acc = get_max(max_test_acc, test_acc)
                log('test#2', nepoch, niter, test_loss, min_test_loss, 
                    test_acc, max_test_acc)
                acc_plotter:add{['test epoch'] = {niter, test_acc}}
                loss_plotter:add{['test epoch'] = {niter, test_loss}}
            end

            acc_plotter:plot()
            loss_plotter:plot()
        end

        -- check point 
        if opt.check_piont and nepoch % opt.check_point == 0 then
            local cp_name = get_log_name() .. '.epoch' .. nepoch
            cp_name = paths.concat(opt.cp_dir, cp_name)
            torch.save(cp_name, {opt, model})
            print('save model to ' .. cp_name)
        end

        -- meet max_epoch ?
        if opt.max_epoch and nepoch >= opt.max_epoch then
            break
        end
    end
    end
    local status, err = pcall(loop)
    if not status then
        if string.find(err, 'interrupted!') and opt.cp_dir then
            local save_path = paths.concat(
                opt.cp_dir, gen_log_name() .. '.interrupted')
            print('\rsave model to ' .. save_path)
            torch.save(save_path, model)
        else
            print(err)
        end
    end
    if not opt.quiet then
        print('Optimization done.')
    end
end
