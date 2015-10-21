require 'nn'

-- Generate a sentenceCNN module perform 1D convolustion
-- D: the dimentions of word embedding .
-- parameters: table of L tables, i-th subtable contain the size of i-th layer, which formatted as {conv_width, conv_stride, kernel_count, pool_width, pool_stride}
function sentenceCNN(D, parameters)
	assert(D > 0)
	assert(#parameters >= 1)
	table.insert(parameters, 1, {0, 0, D, 0, 0})
	net = nn.Sequential()
	for i = 2, #parameters do
		net:add(nn.TemporalConvolution( parameters[i-1][3],
						parameters[i][3],
						parameters[i][1],
						parameters[i][2]))
		net:add(nn.ReLU(true))
		net:add(nn.TemporalMaxPooling( parameters[i][4],
					       parameters[i][5]))
	end
	return net
end
