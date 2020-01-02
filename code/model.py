'''
model.py

Description: Define the model architecture

Date Created: January 02, 2020

Revisions:
  - Jan 02, 2020: Multiple revisions, see below
      > Migrate the plaut_net class from plaut_model.ipynb
      > Add weight initialization (credits to Ian Miller for choosing optimal initalization values)

'''
import torch
import torch.nn as nn

class plaut_net(nn.Module):
    def __init__(self):
        super(plaut_net, self).__init__()
        self.layer1 = nn.Linear(105, 100)
        self.layer2 = nn.Linear(100, 61)
        
    def init_weights(self):
        initrange = 0.1

        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.uniform_(-1.85, -1.85)
        
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.layer2.bias.data.uniform_(-1.85, -1.85)
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        return torch.sigmoid(self.layer2(x))
