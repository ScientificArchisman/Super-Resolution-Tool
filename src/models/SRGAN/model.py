from blocks import Generator_Residual_Block, Discriminator_Block, Generator_Upsample_Block
import torch.nn as nn


class Generator(nn.Module):
    """Generator network for SRGAN"""
    def __init__(self, in_channels, num_upsample_blocks, num_residual_blocks):
        super(Generator, self).__init__()

        ################# Part 1 of Network ###############
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels = 64, kernel_size=9, stride = 1)
        self.activation1 = nn.PReLU(64)

        ################# Part 2 of Network ###############
        # Sequential network with `num_residual_block` resudual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(Generator_Residual_Block(input_channels=64, output_channels=64,
                                                        kernel_size=3, stride=1, padding=1))
        self.res_blocks = nn.Sequential(*res_blocks)

        ################# Part 3 of Network ###############
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        ################# Part 4 of Network ###############
        # Upsampling network
        upsample_blocks = []
        for _ in range(num_upsample_blocks):
            upsample_blocks.append(Generator_Upsample_Block(input_channels=64, output_channels=256,
                                                            kernel_size=3, stride=1, padding=1))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)


        ################# Output Layer ###############
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 3, 
                               kernel_size = 9, stride = 1, padding = 4)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)
        residual = out # Save residual for later

        out = self.res_blocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.upsample_blocks(out)
        out = self.conv3(out)
        return out
    

class Discriminator(nn.Module):
    """ Discriminator network for SRGAN """
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        ################# Part 1 of Network ###############
        self.conv1 = nn.Conv2d(in_channels = input_channels,
                                 out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.activation1 = nn.LeakyReLU(0.2)

        ################# Part 2 of Network ###############
        self.downsample_blocks = []
        self.downsample_blocks.append(
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
                                kernel_size=3, stride=2, padding=1))
        
        self.downsample_layer = nn.Sequential(*self.downsample_blocks)
        
        ################# Part 3 of Network ###############
        self.linear1 = nn.Linear(512*16*16, 1024)
        self.activation2 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.downsample_layer(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.activation2(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out