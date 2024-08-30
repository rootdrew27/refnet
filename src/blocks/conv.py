import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.l1 = Conv()