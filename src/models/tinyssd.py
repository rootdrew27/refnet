import torch
import torch.nn as nn
from src.utils.modules import get_backbone, get_head
from .helpers import make_anchors

class TinySSD(nn.Module):
    def __init__(self, num_classes, imgsz:tuple[int], sizes:list[list[float]], ratios:list[list[float]], device:str):
        """
        `_summary_`

        Args:
            imgsz (tuple[int]): The height and width of the input image
            sizes (list[list[float]]): A list of sizes lists. NOTE: The last sizes list corresponds to the last feature map. 
            ratios (list[list[float]]): A list of ratio lists. NOTE: The last ratio list corresponds to the last feature map. 
        """
        super().__init__()
        self.nc = num_classes
        self.backbone = get_backbone(name='VGG16', load_pretrained=True).to(device)        

        self.anchors, self.num_anchors = make_anchors(self.backbone, sizes, ratios, imgsz, device)

        self.head1 = get_head(task='detect', nc=self.nc, na=self.num_anchors[0]).to(device)
        self.head2 = get_head(task='detect', nc=self.nc, na=self.num_anchors[1]).to(device)
              
    def forward(self, x):
        x1, x2 = self.backbone(x)
        y1, y2 = self.head1(x1), self.head2(x2)
        return y1, y2

            
