import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.vgg import VGG16_Weights
from src.utils.weights import decimate

class VGG16(nn.Module):
    
    def __init__(self, load_pretrained=False):
        super().__init__()        

        self.b1_c1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.b1_c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b1_p1 = nn.MaxPool2d(2, 2) # downsample by (2)
        
        self.b2_c1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.b2_c2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.b2_p1 = nn.MaxPool2d(2, 2) # downsample by (2)
        
        self.b3_c1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.b3_c2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b3_c3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b3_p1 = nn.MaxPool2d(2, 2, ceil_mode=True) # downsample by 2 (potentially adding 1 afterward)
        
        self.b4_c1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.b4_c2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.b4_c3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.b4_p1 = nn.MaxPool2d(kernel_size=2, stride=2) # downsample by 2

        self.b5_c1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.b5_c2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.b5_c3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.b5_p1 = nn.MaxPool2d(3, 1, 1)
        
        self.b6_c1 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6) 
        self.b6_c2 = nn.Conv2d(1024, 1024, 1)
        
        if load_pretrained: self.load_pretrained_weights()
        
    def forward(self, x):
        out = F.relu(self.b1_c1(x))
        out = F.relu(self.b1_c2(out))
        out = self.b1_p1(out)
        
        out = F.relu(self.b2_c1(out))
        out = F.relu(self.b2_c2(out))
        out = self.b2_p1(out)
        
        out = F.relu(self.b3_c1(out))
        out = F.relu(self.b3_c2(out))
        out = F.relu(self.b3_c3(out))
        out = self.b3_p1(out)
        
        out = F.relu(self.b4_c1(out))
        out = F.relu(self.b4_c2(out))
        fmap1 = out = F.relu(self.b4_c3(out)) # 512, H/8 + 1?, W/8 + 1?
        out = self.b4_p1(out)
        
        out = F.relu(self.b5_c1(out))
        out = F.relu(self.b5_c2(out))
        out = F.relu(self.b5_c3(out))
        out = self.b5_p1(out)
        
        out = F.relu(self.b6_c1(out))
        fmap2 = F.relu(self.b6_c2(out)) # 1024, H/16 (+1)?, W/16 (+1)?
        
        return fmap1, fmap2
    
    def load_pretrained_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        
        pretrained_state_dict = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of b6_c1 and b6_c2
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['b6_c1.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['b6_c1.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['b6_c2.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['b6_c2.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)        
                
        self.load_state_dict(state_dict)
        
        print('\nLoaded Pretrianed weights for VGG16')