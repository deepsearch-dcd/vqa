local util = require 'util/util'

-- wrap the dataset comes from COCOQA.load_data(), adding functions as follows:
-- size(), reset(), next(), cuda()
--
-- arguments:
-- dataset          dataset comes from COCOQA.load_data()
-- imageFeatures    assemble it to dataset.images
-- [disability]     used to some disability model. ='blind' next() return (Q,A);
--                  ='deaf' next() return (V,Q); not given, next() return ({V,Q},A).
-- [cacheFeature]    if not given, don't aeemble dataset.images as a whole and look
--                  up each a time in next().
local function COCODatasetWrapper(dataset, imageFeatures, disability, cacheFeature)
    assert(dataset)
    assert(imageFeatures)

    local Tensor = torch.Tensor
    dataset.images = Tensor(dataset.images)
    for i, q in ipairs(dataset.questions) do
        dataset.questions[i] = Tensor(q)
    end
    dataset.answers = Tensor(dataset.answers)

    if cacheFeature then
        dataset.images = util.assemble(dataset.images, imageFeatures)
    end

    function dataset:size()
        return self.nsample
    end

    dataset.current_index = 0

    function dataset:reset()
        self.current_index = 0
    end

    function dataset:_next()
        self.current_index = self.current_index + 1
        local index = self.current_index
        if index > self:size() then
            return nil
        end
        return self.images[index], self.questions[index], self.answers[index]
    end
    assert((not disability) or (disability == 'blind') 
            or (disability == 'deaf'))
    if disability == 'blind' then
        function dataset:next()
            local _, Q, A = dataset:_next()
            return Q, A
        end
    elseif cacheFeature then
        if not disability then
            function dataset:next()
                local V, Q, A = dataset:_next()
                return {V, Q}, A
            end
        elseif disability == 'deaf' then
            function dataset:next()
                local V, _, A = dataset:_next()
                return V, A
            end
        end
    else
        if not disability then
            function dataset:next()
                local V, Q, A = dataset:_next()
                V = imageFeatures[V]
                if self.cuda then
                    V = V:cuda()
                end
                return {V, Q}, A
            end
        elseif disability == 'deaf' then
            function dataset:next()
                local V, Q, A = dataset:_next()
                V = imageFeatures[V]
                if self.cuda then
                    V = V:cuda()
                end
                return V, A
            end
        end
    end

    function dataset:cuda()
        self.images = self.images:cuda()
        for i, q in ipairs(self.questions) do
            self.questions[i] = q:cuda()
        end
        self.answers = self.answers:cuda()
        self.cuda = true
    end
end

return COCODatasetWrapper
