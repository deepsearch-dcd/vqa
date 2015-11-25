--[[
Usage:

require 'util/Plotter'
local plotter = Plotter('.', 'acc') -- the plot will be saved in `./`
plotter:setNames('train', 'test') -- give each curve a name
plotter:add{        -- add points to the curve
    ['train'] = {iter, train_loss},
    ['test'] = {iter, test_loss}
}
plotter:plot()  -- plot all the curves to "[tag]_filename.svg", you can set a specific tag by modifing `plotter.tag`. you can also give the second argment in the form of table, to specific the names of curve you want to plot.
----]]

require 'paths'
require 'gnuplot'

local Plotter = torch.class('Plotter')

function Plotter:__init(dirname, figname)

    self.dirname = assert(dirname)  -- save the figure file
    os.execute('mkdir -p "' .. self.dirname .. '"')

    if figname then     -- the name of figure
        self.name = figname
    else
        self.name = 'default'
    end

    self.plots = {}
    self.styles = {}
    self.defaultStyle = '-'
    self.tag = 'default'
end

function Plotter:setFigname(name)
    assert(name)
    self.name = name
end

function Plotter:setNames(...)
    local names = {...}
    for _, name in ipairs(names) do
        if self.plots[name] == nil then
            self.plots[name] =  {x = {}, y = {}}
            self.styles[name] = self.defaultStyle
        else
            error('[Plotter]: the name ' .. name .. ' already exists!')
        end
    end
end

function Plotter:__nameExistsOrError(name)
    if self.plots[name] == nil then
        error('[Plotter]: unknown name ' .. name)
    end
end

function Plotter:add(points)
    for name, point in pairs(points) do
        self:__nameExistsOrError(name)
        table.insert(self.plots[name].x, point[1])
        table.insert(self.plots[name].y, point[2])
    end
end

function Plotter:style(symbols)
    for name, style in pairs(symbols) do
        self:__nameExistsOrError(name)
        self.styles[name] = style
    end
end

function Plotter:plot(names)
    local svgfile = paths.concat(self.dirname, self.tag .. '_' .. self.name .. '.svg')
    local plots = {}
    local plotit = false
    local add_plot = function(name)
        self:__nameExistsOrError(name)
        if #self.plots[name].x > 0 then
            local x = torch.Tensor(self.plots[name].x)
            local y = torch.Tensor(self.plots[name].y)
            table.insert(plots, {name, x, y, self.styles[name]})
            plotit = true
        end
    end

    if names then
        for _, name in ipairs(names) do
            add_plot(name)
        end
    else
        for name, _ in pairs(self.plots) do
            add_plot(name)
        end
    end

    if plotit then
        os.execute('rm -f "' .. svgfile .. '"')
        local fig = gnuplot.svgfigure(svgfile)
        gnuplot.plot(plots)
        gnuplot.grid('on')
        gnuplot.title(self.name)
        gnuplot.plotflush()
        gnuplot.close(fig)
    end

end
