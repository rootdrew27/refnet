import torch
import torch.nn as nn

class Detect(nn.Module):
    def __init__(self, nc, na):
        super(Detect, self).__init__()
        # Class Predicition
        self.cp_c1 = nn.LazyConv2d((nc + 1), 3, 1, 1) # NOTE : Consider multiplying (nc + 1) by na to allow each anchor to predict its own classes
                
        # Bbox Predicition
        self.bp_c1 = nn.LazyConv2d(4 * na, 3, 1, 1)
        
    def forward(self, x):
        return self.cp_c1(x), self.bp_c1(x)