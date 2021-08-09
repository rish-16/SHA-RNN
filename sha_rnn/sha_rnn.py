import torch
from torch import nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, T, d):
        super().__init__()
        self.T = T
        self.d = d
        
        self.W = nn.Linear(d, d)
        self.U = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.b = nn.Parameter(torch.rand(1, d))
        
    def forward(self, x, h_t):
        y_t = None
        for t in range(self.T):
            a_t = self.b + (self.W(h_t)) + (self.U(x[t]))
            h_t = torch.tanh(a_t)
            o_t = self.V(h_t)
            y_t = F.softmax(o_t, 1)
        return y_t, h_t
    
    def init_hidden(self):
        return torch.zeros(1, self.d)
    
class DecoderRNN(nn.Module):
    def __init__(self, h, output_size):
        super().__init__()
        self.h = h
        self.embedding = nn.Embedding(output_size, h)
        self.gru = nn.GRU(h, h)
        self.out = nn.Linear(h, output_size)

    def forward(self, x, h): 
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, h = self.gru(output, h)
        output = F.softmax(self.out(output[0]), 1)
        return output, h
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.d)