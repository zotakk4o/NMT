import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")


batchSize = 32
wordEmbSize = 512

hidSize = 512
dropout = 0.5

layers = 2

uniform_init = 0.1
learning_rate = 0.001
clip_grad = 5
learning_rate_decay = 0.5

maxEpochs = 20
log_every = 250
test_every = 1000

max_patience = 2
max_trials = 10