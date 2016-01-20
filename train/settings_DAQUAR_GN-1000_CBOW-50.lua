require 'nn'

dataset = 'DAQUAR'

feature_path = 'feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy'

embedding_path = 'word_embedding/CBOW_50.t7'

criterion = nn.ClassNLLCriterion()

opt = {
    batch_size = 32,
    -- display_interval = 500,
    learningRate = 0.001,
    weightDecay = 0.0005,
    momentum = 0.9,
    gpuid = 0,
    tag = 'default',
}
