import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model_parts import *

class Discriminator(nn.Module):
    def __init__(self, in_channels, first_out_channels, n_layers = 3):
        super().__init__()
        self.first_out_channels = first_out_channels
        seq = [nn.Conv3d(in_channels, first_out_channels, kernel_size=4, 
                         padding=2, stride=2),
               nn.LeakyReLU(0.2, True)]
        filt_mult = 1
        prev_mult = 1
        for i in range(1,n_layers):
            prev_mult = filt_mult
            filt_mult = min(2**i, 8)
            seq += [nn.Conv3d(first_out_channels*prev_mult, first_out_channels*filt_mult,
                          kernel_size=4, stride=2, padding=2, bias=False),
                    nn.BatchNorm3d(first_out_channels*filt_mult),
                    nn.LeakyReLU(0.2, True)]
            
        prev_mult = filt_mult
        filt_mult = min(2**n_layers, 8)
        
        seq += [nn.Conv3d(first_out_channels*prev_mult, first_out_channels*filt_mult,
                      kernel_size=4, stride=1, padding=2, bias=False),
                nn.BatchNorm3d(first_out_channels*filt_mult),
                nn.LeakyReLU(0.2, True)]
        
        # This is the output layer at last, hence only 1 filter
        seq += [nn.Conv3d(first_out_channels*filt_mult, 1, 
                          kernel_size=4, stride=1, padding=2)]
        
        self.model = nn.Sequential(*seq)
            
    def forward(self, x):
        return self.model(x)
    

# The example has multiple start convolutions before downsampling;
# worth trying?
class Generator1(nn.Module): # 3D UNet with doubleconvs
    def __init__(self, in_channels, first_out, n_down):
        # n_down is the total number of DoubeConv blocks before we start
        # going up; hence the count excludes inconv
        # Think of it as the # of down arrows on a typical U-net graph,
        # which is equivalent to the number of time Max Pooling occurs
        assert isinstance(first_out, int)
        super().__init__()
        self.in_channels = in_channels
        self.first_out = first_out
        self.n_down = n_down

        self.inconv = DoubleConv(self.in_channels, first_out)
        
        # Construct an arbitrary-depth U-net 
        for i in range(1,n_down+1):
            k_down = 2**(i-1)
            setattr(self, f'down{i}', DownBlock(first_out*k_down, first_out*(k_down*2)))
        for i in range(1,n_down+1):
            k_up = 2**(self.n_down-i+1)
            setattr(self, f'up{i}', UpBlock(first_out*k_up, first_out*(k_up//2)))
            
                    
            
        self.outconv = nn.Conv3d(first_out, 1, 3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Iterate through the down's and save their outputs in a list
        # Then add that list to each up in REVERSE ORDER
        resids = []
        
        resids.append(self.inconv(x))
        for i in range(1,self.n_down+1):
            #print(resids[i-1][-1])
            # Use previous resid as input to each
            resids.append(getattr(self, f'down{i}')(resids[i-1]))
            
        out = resids[-1]
        # The very botton is not used as a skip connection
        # so index with -(i+1) to ignore it
        for i in range(1, self.n_down+1):
            out = getattr(self, f'up{i}')(out, resids[-(i+1)])
            #out = exec(f'self.up{i}({out}, {resids[-(i+1)]})')
            
        out = self.outconv(out)
        return out
    
class Generator2(nn.Module): # Uses single conv with stride=2 for downsampling
    def __init__(self, in_channels, first_out_channels, n_layers = 3):
        super().__init__()
        self.in_channels = in_channels
        self.first_out = int(first_out_channels)
        self.n_down = n_layers
        
        self.inconv = nn.Conv3d(in_channels, first_out_channels, kernel_size=3,
                                stride=1, padding=1)
        
        for i in range(1,n_layers+1):
            k_down = int(2**(i-1))
            setattr(self, f'down{i}', nn.Sequential(
                nn.Conv3d(self.first_out*k_down, self.first_out*(k_down*2),
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(self.first_out*(k_down*2)),
                nn.LeakyReLU(0.2, True)
            )
                   )
            
        for i in range(1,n_layers+1):
            k_up = 2**(self.n_down-i+1)
            setattr(self, f'up{i}', UpBlock(self.first_out*k_up, self.first_out*(k_up//2), 2))
            
        self.outconv = nn.Conv3d(self.first_out, 1, 3, padding=1)
            
    def forward(self, x):
        resids = []
        
        resids.append(self.inconv(x))
        #print(resids[0].shape)
        for i in range(1,self.n_down+1):
            #print(resids[i-1][-1])
            # Use previous resid as input to each
            resids.append(getattr(self, f'down{i}')(resids[i-1]))
            #print(resids[i].shape)
            
        out = resids[-1]
        # The very botton is not used as a skip connection
        # so index with -(i+1) to ignore it
        for i in range(1, self.n_down+1):
            out = getattr(self, f'up{i}')(out, resids[-(i+1)])
            #print(out.shape)
            
        out = self.outconv(out)
        #print(out.shape)
        return out
                    
def get_models(d_first_out, d_n_layer, g_first_out, g_n_layer, gen_type = 1):
    if gen_type == 1:
        return Discriminator(1, d_first_out, d_n_layer), Generator1(9, g_first_out, g_n_layer)
    else:
        return Discriminator(1, d_first_out, d_n_layer), Generator2(9, g_first_out, g_n_layer)