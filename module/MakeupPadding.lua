require 'nn'
local MakeupPadding, parent = torch.class('nn.MakeupPadding', 'nn.Module')

function MakeupPadding:__init(dim, len, nInputDim, value)
    self.value = value or 0
    self.dim = dim
    self.len = len
    self.nInputDim = nInputDim
    self.outputSize = torch.LongStorage()
    parent.__init(self)
end

function MakeupPadding:updateOutput(input)
    local dim = self.dim
    if self.nInputDim and input:dim() ~= self.nInputDim then
        dim = dim + 1
    end
    local len = self.len
    local value = self.value
    self.outputSize:resize(input:dim())
    self.outputSize:copy(input:size())
    self.outputSize[dim] = len
    self.output:resize(self.outputSize)
    if input:size(dim) <= len then
        self.output:fill(value)
        self.output:narrow(dim, 1, input:size(dim)):copy(input)
    else
        self.output:copy(input:narrow(dim, 1, len))
    end
    return self.output
end

function MakeupPadding:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    local dim = self.dim
    if self.nInputDim and input:dim() ~= self.nInputDim then
        dim = dim + 1
    end
    if input:size(dim) <= gradOutput:size(dim) then
        self.gradInput:copy(gradOutput:narrow(dim, 1, input:size(dim)))
    else
        self.gradInput:fill(0)
        self.gradInput:narrow(dim, 1, gradOutput:size(dim)):copy(gradOutput)
    end
    return self.gradInput
end
