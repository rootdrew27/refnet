from src.backbones.vgg import VGG16
from src.blocks.detect import Detect

# TODO create this dict by travesing the backbones folder 
backbones = {
    'VGG16': VGG16
} 

def get_backbone(name, load_pretrained=True):
    """
    Options include: 'VGG16'

    Args:
        name (str): The name of the class  
        load_pretrained (bool, optional): Load weights of a pretrained model if available. Defaults to True.

    Returns:
        _type_: _description_
    """
    bbone_class = backbones[name]
    bbone = bbone_class(load_pretrained=load_pretrained)
    return bbone


# TODO create this dict by travesing the head folder 
tasks = {
    'detect': Detect
}

def get_head(task, nc=None, na=None):
    """
    Args:
        task (str): The task to be performed by the head
        nc (int): The number of classes
        na (int): The number of anchors

    Returns:
        nn.Module: a head
    """
    head_class = tasks[task]
    head = head_class(nc, na)
    return head
    