import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapePrinter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        print(x.size())
        return x
    
class Thrower(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        raise InterruptedError
        return x
        
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

#TODO: Make model
class Simple_VAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        return x  
