local Freeze, parent = torch.class('nn.Freeze', 'nn.Module')

function Freeze:__init(module)
    self.module = module
    parent.__init(self)
end

function Freeze:updateOutput(input)
    self.output = self.module:updateOutput(input)
    return self.output
end

function Freeze:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end
