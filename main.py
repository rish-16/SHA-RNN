import torch
from sha_rnn import EncoderRNN

x = torch.rand(10, 128)
encoder = EncoderRNN(10, 128)
hidden = encoder.init_hidden()
y, hidden = encoder(x, hidden)

print (y.shape)
print (hidden.shape)