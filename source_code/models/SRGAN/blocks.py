import torch.nn as nn 


class Generator_Residual_Block(nn.Module):
    """ Generator Residual Block 
    Conv -> BN -> PReLU -> Conv -> BN -> Add"""
    def __init__(self, input_channels = 64, 
                 output_channels = 64, 
                 kernel_size = 3, stride = 1, padding = 1):
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
    

class Discriminator_Block(nn.Module):
    """ Discriminator  Block
    Conv -> BN -> LeakyReLU"""
    def __init__(self, input_channels, output_channels,
                 kernel_size, stride, padding):
        super(Discriminator_Block, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 
                          kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.activation(output)

        return output
    


class UpsampleBlock(nn.Module):
    """ Upsample block using PixelShuffle
    Conv -> PixelShuffle -> PReLU """
    def __init__(self, in_channels, up_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_factor ** 2, kernel_size=3, 
                              stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x