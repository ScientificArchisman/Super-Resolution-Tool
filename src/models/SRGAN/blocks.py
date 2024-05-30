import torch 
import torch.nn as nn 
import numpy as np 


class Generator_Residual_Block(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 kernel_size, stride, padding):
        super(Generator_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 
                          kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.activation = nn.PReLU(output_channels) 
        self.conv2 = nn.Conv2d(output_channels, output_channels, 
                          kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)   
        out = self.bn2(out)
        out += residual
        return out
    

class DisCriminator_Residual_Block(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size, stride, padding):
        super(DisCriminator_Residual_Block, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 
                          kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(sel)