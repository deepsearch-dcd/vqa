if vqalstm==nil then
	require('..')
end

local inputs
local outputs

local input_size = 30
local rnn_size = 20
local num_layer = 2

local input = torch.rand(3, input_size)

-- initialize LSTM model
local lstm_config = {
  in_dim = input_size,
  mem_dim = rnn_size,
  num_layers = num_layer,
  gate_output = true,
}

local lstm = vqalstm.LSTM(lstm_config)

local lstm_output = lstm:forward(input)
print(num_layer ..' layers vqalstm.LSTM output:')
print(lstm_output)