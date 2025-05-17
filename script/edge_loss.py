import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F



class EdgeConvloss(nn.Module):
    def __init__(self):
        super(EdgeConvloss, self).__init__()
        kernel = torch.tensor([[-1., -1., -1.], 
                               [-1.,  8., -1.], 
                               [-1., -1., -1.]])
        self.kernel = kernel.view(1, 1, 3, 3)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        device = input.device
        self.kernel = self.kernel.to(device)
        edge1 = F.conv2d(input, self.kernel, padding=1)
        edge2 = F.conv2d(target, self.kernel, padding=1)
        return F.mse_loss(edge1, edge2)
    

def edgeconvloss(input: Tensor, target: Tensor):
    kernel = torch.tensor([ [-1., -1., -1.], 
                            [-1.,  8., -1.], 
                            [-1., -1., -1.]]).view(1,1,3,3)
    edge1 = F.conv2d(input, kernel, padding=1)
    edge2 = F.conv2d(target, kernel, padding=1)
    return F.mse_loss(edge1, edge2)