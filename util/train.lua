require 'optim'
require 'paths'

require 'util/Plotter'
local util = require 'util/util'

-- wrap a batch view to dataset
function batch_view(dataset, batch_size, use_cuda, blind)
    local new_dataset = {
        questions={}, 
        answers={},
        nvocab = dataset.nvocab,
        nanswer = dataset.nanswer,
    }
    if not blind then
        new_dataset.images = {}
        new_dataset.nimage = dataset.nimage
    end
    local nsample = dataset.questions:size(1)
    local new_idx = 0
    for old_idx = 1, nsample, batch_size do
        local from = old_idx
        local to = math.min(old_idx + batch_size - 1, nsample)
        new_idx = new_idx + 1
        if use_cuda then
            if not blind then
                new_dataset.images[new_idx] = dataset.images[{{from, to}}]:float():cuda()
            end
            new_dataset.questions[new_idx] = dataset.questions[{{from, to}}]:float():cuda()
            new_dataset.answers[new_idx] = dataset.answers[{{from, to}}]:float():cuda()
        else
            if not blind then
                new_dataset.images[new_idx] = dataset.images[{{from, to}}]
            end
            new_dataset.questions[new_idx] = dataset.questions[{{from, to}}]
            new_dataset.answers[new_idx] = dataset.answers[{{from, to}}]
        end
    end
    new_dataset.nsample = new_idx
    return new_dataset
end

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
    local LOG_FORMAT = '(%s)Epoch %d, Iteration %d, loss = %.2f(%.2f), acc = %.2f(%.2f)'
    print(string.format(LOG_FORMAT, tag, epoch, iter, loss, min_loss, acc, max_acc))
end

-- evaluate model on the given testset
local function test(model, criterion, testset, batch_mode, blind)
    
    -- initialize confusion matrix
    local confusion = optim.ConfusionMatrix(testset.nanswer)
    local confusion_add = confusion.add
    if batch_mode then
        confusion_add = confusion.batchAdd
    end

    -- initialize total loss
    local total_loss = 0

    -- loop testset
    for i = 1, testset.nsample do
        -- get data
        local x = 1  -- placeholder
        if not blind then
            x = {testset.images[i], testset.questions[i]}
        else
            x = testset.questions[i]
        end
        local t = testset.answers[i]
        local loss = criterion:forward(model:forward(x), t)
        confusion_add(confusion, model.output, t)
        total_loss = total_loss + loss
    end

    confusion:updateValids()
    return total_loss/testset.nsample, confusion.totalValid
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

--[[
   train the given model

   `dataset` must has three keys: images, questions, answers.
   (images[1], questions[1], answers[1]) as a sample.

    `opt.max_epoch`         option
    `opt.batch_size`        option[1]
    `opt.log_dir`           option
    `opt.learningRate`      option[0.001]
    `opt.weightDecay`       option[0.0005]
    `opt.momentum           option[0.9]
    `opt.display_interval`  option
    `opt.gpuid`             option[0]
    `opt.plot_dir`          option
    `opt.tag`               option              distinguish different training
    `opt.quiet`             option[false]
    `opt.check_point`       option
    `opt.cp_dir`            option
    `opt.blind`             option
--]]
function train(opt, model, criterion, trainset, testset)

    -- fill `opt` with some default value
    opt.learningRate = opt.learningRate or 1e-3
    opt.weightDecay = opt.weightDecay or 5e-4
    opt.momentum = opt.momentum or 0.9
    opt.gpuid = opt.gpuid or -1
    opt.plot_dir = opt.plot_dir or 'done'
    opt.tag = opt.tag or 'default'
    opt.cp_dir = opt.cp_dir or '.'

    -- trigger loging
    if opt.log_dir then
        local cmd = torch.CmdLine()
        local log_name = gen_log_name()
        cmd:log(paths.concat(opt.log_dir, log_name), opt)
        cmd:addTime('vqa', '%F %T')
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
    end

    -- if in gpu mode, convert model and criterion to cuda type
    if opt.gpuid >= 0 then
        model = model:cuda()
        criterion = criterion:cuda()
    end

    -- get parameters and gradients from model
    local parameters, gradParameters = model:getParameters()

    -- split dataset in batch mode and convert data to cuda type
    if opt.batch_size and opt.batch_size > 1 then
        trainset = batch_view(trainset, opt.batch_size, opt.gpuid>=0, opt.blind)
        if testset then
            testset = batch_view(testset, opt.batch_size, opt.gpuid>=0, opt.blind)
        end
        collectgarbage()
    elseif opt.gpuid >= 0 then
        util.to_cuda(trainset)
        if testset then
            util.to_cuda(testset)
        end
    end
    local iters_per_epoch = trainset.nsample   

    -- configure optim train state
    local config = {}
    config.learningRate = opt.learningRate
    config.weightDecay = opt.weightDecay
    config.momentum = opt.momentum


    -- initialize some variables used to control the training loop
    
    -- the total epoch and iteration util now
    local nepoch, niter = 0, 0
    -- the interval of iterations to display the result of evaluation
    local display_interval = opt.display_interval
    -- accumulated loss for the interval of the iteration and epoch
    local iter_loss, epoch_loss = 0, 0
    -- statistic for evaluation
    local min_train_loss, min_test_loss, max_train_acc, max_test_acc = 1e12,1e12,0,0
    -- save old parameters to compute the diff for debug
    -- local old_params = torch.Tensor()

    -- evaluation tools
    local iter_confusion = optim.ConfusionMatrix(trainset.nanswer)
    local epoch_confusion = optim.ConfusionMatrix(trainset.nanswer)
    local confusion_add = iter_confusion.add
    if opt.batch_size and opt.batch_size > 1 then
        confusion_add = iter_confusion.batchAdd
    end

    -- plot tools
    local acc_plotter = get_plotter(opt.plot_dir, 'acc', opt.tag)
    local loss_plotter = get_plotter(opt.plot_dir, 'loss', opt.tag)

    -- prepare for check point
    if opt.check_point then
        os.execute('mkdir -p "' .. opt.cp_dir .. '"')
    end

    -- loop epoch
    while true do

        -- current epoch
        nepoch = nepoch + 1

        -- epoch start time
        --local time = sys.clock()

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
                
                -- get data
                -- x, t = unpack(trainset[i])
                local x = 1 -- placeholder
                if not opt.blind then
                    x = {trainset.images[i], trainset.questions[i]}
                else
                    x = trainset.questions[i]
                end
                local t = trainset.answers[i]

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
            
            -- save initial parameters
            --if nepoch == 1 and niter == 1 then
            --    old_params = parameters:clone()
            --end

            -- optimize on current mini-batch
            optim.sgd(feval, parameters, config)

            -- display the evaluation of the last interval
            if not opt.quiet and display_interval and niter % display_interval == 0 then
                iter_confusion:updateValids()
                local train_loss = iter_loss / display_interval
                log('train#1', nepoch, niter, train_loss, 0, iter_confusion.totalValid, 0)
                
                acc_plotter:add{['train iteration'] = {niter, iter_confusion.totalValid}}
                loss_plotter:add{['train iteration'] = {niter, train_loss}}
                
                --acc_plotter:plot()
                --loss_plotter:plot()
                
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
            log('train#2', nepoch, niter, train_loss, min_train_loss, epoch_confusion.totalValid, max_train_acc)
            acc_plotter:add{['train epoch'] = {niter, epoch_confusion.totalValid}}
            loss_plotter:add{['train epoch'] = {niter, train_loss}}
            epoch_loss = 0
            epoch_confusion:zero()

            if testset then
                -- display testset evaluation result
                local test_loss, test_acc = test(model, criterion, testset, opt.batch_size and opt.batch_size > 1, opt.blind)
                min_test_loss = get_min(min_test_loss, test_loss)
                max_test_acc = get_max(max_test_acc, test_acc)
                log('test#2', nepoch, niter, test_loss, min_test_loss, test_acc, max_test_acc)
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

        -- epoch end time
        --time = sys.clock() - time
        --print('time to run a epoch = ' .. (time*1000) .. 'ms')

        -- meet max_epoch ?
        if opt.max_epoch and nepoch >= opt.max_epoch then
            break
        end
    end
    if not opt.quiet then
        print('Optimization done.')
    end
end
