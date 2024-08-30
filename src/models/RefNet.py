import torch
import torch.nn as nn
import torch.nn.functional as F

from d2l import torch as d2l

import numpy as np

from src.utils.modules import get_backbone, get_head
from .helpers import get_feat_map_sizes

class RefNet(nn.Module):
    def __init__(self, imgsz:tuple[int]|int,  asp_ratios=[0.5, 1.0, 2.0], nc=1):
        super(RefNet, self).__init__()
        self.nc = nc
        
        self.backbone = get_backbone('VGG16', pretrained=True)
        self.head1 = get_head(task='detect', nc=1, na=3)        
        self.head2 = get_head(task='detect', nc=1, na=4)
       
        self.head3 = get_head(task='detect', nc=1, na=3)
        self.head4 = get_head(task='detect', nc=1, na=3)
        self.head5 = get_head(task='detect', nc=1, na=3)
        
        feat_map_sizes = get_feat_map_sizes(self.backbone, imgsz)
        
        self.priors = anchor.bxs(imgsz, feat_map_sizes, asp_ratios, named=False)
        

    def forward(self, x_cur, x_ref):
        pass

