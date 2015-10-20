require 'paths'
local util = require 'util/util'

do
for f in paths.files("test") do
	if util.start_with(f, 'test_') then
		dofile('test/'..f)
	end
end
end
