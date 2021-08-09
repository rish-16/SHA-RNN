import torch
from sha_rnn import DecoderRNN

x = torch.rand(15, 128)
encoder = DecoderRNN(15, 128)
hidden = encoder.init_hidden()
y, hidden = encoder(x, hidden)

print (y.shape)
print (hidden.shape)