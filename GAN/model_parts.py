import torch
from torch import nn, optim
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.model(x)
    
# A DownBlock in UNet is a simple Double Convolution followed by 2x2 maxpooling
# We include maxpooling at the start since it isn't at the end of the DoubleConv class
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            #nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )
        
    def forward(self, x):
        return self.model(x)
    
# Basically a downblock with transpose conv instead, but gets joined
# by output of corresponding downblock (mirror-image, across the 'U' of UNet)
# after Upsampling but before DoubleConv

# Check OG paper that kernel_size, stride need to be 2. 
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gen_type=1):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels//2,
                                           kernel_size=2, stride=2)
        self.gen_type = gen_type
        if gen_type == 1:
            self.doubleconv = DoubleConv(in_channels, out_channels)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
            self.bn = nn.BatchNorm3d(out_channels)
            self.activate = nn.LeakyReLU(0.2, True)
        
    def forward(self, x, x_skip):
        x = self.upsample(x)
        
        # x is NCDHW
        d_diff = x_skip.size()[2] - x.size()[2]
        h_diff = x_skip.size()[3] - x.size()[3]
        w_diff = x_skip.size()[4] - x.size()[4]
        
        # pad works backwards, i.e. it will first pad the LAST dimension, then
        # second last, etc
        # Specify as [left_pad, right_pad, top_bad, bot_pad, front_pad, back_pad]
        x = F.pad(x, [w_diff // 2, w_diff - w_diff // 2,
                       h_diff // 2, h_diff - h_diff // 2,
                       d_diff // 2, d_diff - d_diff // 2])
        x = torch.cat([x_skip, x], dim=1) # Pad along channels dimension
        if self.gen_type == 1:
            return self.doubleconv(x)
        else:
            x = self.conv(x)
            return self.activate(self.bn(x))