# file to train RefNet
from codetiming import Timer

class RefNetTrainer():
    def __init__(self, model, optimizer, criterion, on_start=[], on_end=[], save_dir=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.losses = []
        # callbacks 
        self.on_start = on_start 
        self.on_end = on_end
        self.save_dir = save_dir
        self.stats = {}
        
    def train(self, dataloader, epochs, device):
        
        print(f"Starting training on {device}")
        
        self.model.to(device)
        
        for i in range(epochs):
            
            with Timer(name='context manager'):
                for j, batch in enumerate(dataloader):
                
        