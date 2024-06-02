from source_code.models.SRGAN.blocks import Generator_Residual_Block, Discriminator_Block, UpsampleBlock
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=16, 
                 num_upsample_blocks=2, upsample_factor=2):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        
        self.residual_blocks = nn.Sequential(
            *[Generator_Residual_Block(64) for _ in range(num_residual_blocks)]
        )
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(64, upsample_factor) for _ in range(num_upsample_blocks)]
        )
        
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out2 = self.residual_blocks(out1)
        out3 = self.bn(self.conv2(out2))
        out4 = out1 + out3
        
        out5 = self.upsample_blocks(out4)
        out6 = self.conv3(out5)
        return out6
    

class Discriminator(nn.Module):
    """ Discriminator network for SRGAN """
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        ################# Part 1 of Network ###############
        self.conv1 = nn.Conv2d(in_channels = input_channels,
                                 out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.activation1 = nn.LeakyReLU(0.2)

        ################# Part 2 of Network ###############
        self.downsample_blocks = [
            Discriminator_Block(input_channels=64, output_channels=64,
                                kernel_size=3, stride=2, padding=1), 
            Discriminator_Block(input_channels=64, output_channels=128,
                                kernel_size=3, stride=1, padding=1),
            Discriminator_Block(input_channels=128, output_channels=128,
                                kernel_size=3, stride=2, padding=1),
            Discriminator_Block(input_channels=128, output_channels=256,
                                kernel_size=3, stride=1, padding=1),
            Discriminator_Block(input_channels=256, output_channels=256,
                                kernel_size=3, stride=2, padding=1),
            Discriminator_Block(input_channels=256, output_channels=512,
                                kernel_size=3, stride=1, padding=1),
            Discriminator_Block(input_channels=512, output_channels=512,
                                kernel_size=3, stride=2, padding=1)]
        
        self.downsample_layer = nn.Sequential(*self.downsample_blocks)
        
        ################# Part 3 of Network ###############
        self.linear1 = nn.Linear(1, 1024)
        self.activation2 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.downsample_layer(out)
        out = out.view(out.size(0), -1)

        # Dynamically calculate the input size for the linear layer
        if self.linear1.in_features != out.size(1):
            self.linear1 = nn.Linear(out.size(1), 1024).to(out.device)

        out = self.linear1(out)
        out = self.activation2(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out