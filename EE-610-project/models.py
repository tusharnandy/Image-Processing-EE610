import torch
from torch import nn


class ResidualDense(nn.Module):
  # Dense Layer = Conv Layer + ReLU
  def __init__(self, input_dim, output_dim):
    super(ResidualDense, self).__init__()
    self.Conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1) 
    self.relu = nn.ReLU()

  def forward(self, x):
    h = self.Conv(x)
    f =  self.relu(h)
    residue = (x, f)
    return torch.cat(residue, 1)


class RDB(nn.Module):
  def __init__(self, input_dim, G, C):
    super(RDB, self).__init__()
    internal_blocks = [ResidualDense(input_dim + G * i, G) for i in range(C)]  # Number of convolution layers inside one RD Block
    self.local_features = nn.Sequential(*internal_blocks)

    # Concatenating all features for LFF
    self.fuse = nn.Conv2d(in_channels=input_dim + G * C, # These are concatenated from previous layers
                            out_channels= G,
                            kernel_size=1)

  def forward(self, x):
    out = self.local_features(x)
    out = self.fuse(out)
    return x + out  # Residual connection


class RDN(nn.Module):
  def __init__(self, num_channels, G_0, G, D, C, n):
    super(RDN, self).__init__()

    self.SF1 = nn.Conv2d(in_channels=num_channels, out_channels=G_0,
                         kernel_size=3, padding=1, stride=1)
    self.SF2 = nn.Conv2d(in_channels=G_0, out_channels=G_0,
                         kernel_size=3, padding=1, stride=1)
    RDB_array = [RDB(G_0, G, C)] + [RDB(G, G, C) for i in range(1, D)]
    self.RD_blocks = nn.ModuleList(RDB_array)
    self.GlobalFusion= nn.Sequential(nn.Conv2d(in_channels = G * D, 
                                   out_channels=G_0,
                                   kernel_size=1,stride=1), 
                                   nn.Conv2d(in_channels = G_0, 
                                   out_channels=G_0,
                                   kernel_size=3,
                                   padding=1,
                                    stride=1)) 
    
    if n == 3:
        self.upconv = nn.Conv2d(G_0, n*G_0*n, 
            kernel_size=3, padding=1, stride=1)
        self.upconv = nn.Sequential(self.upconv, nn.PixelShuffle(3))
    else:
        upconv_array = [nn.Conv2d(G_0, 2*G_0*2, kernel_size=3, padding=1, stride=1), nn.PixelShuffle(2)]
        if n == 4:
            upconv_array += [nn.Conv2d(G_0, 2*G_0*2, kernel_size=3, padding=1, stride=1), nn.PixelShuffle(2)]
        self.upconv = nn.Sequential(*upconv_array)
    self.scaled = nn.Conv2d(G_0, num_channels, kernel_size=3, padding=1, stride=1)
    self.D = D

  def forward(self, x):
    f_ = self.SF1(x)
    shallow_features = self.SF2(f_)

    f = []
    f.append(self.RD_blocks[0](shallow_features))
    for i in range(1, self.D):
        f.append(self.RD_blocks[i](f[i-1]))

    skip_connection = torch.cat(f, 1) 
    x1 = self.GlobalFusion(skip_connection) + f_
    x2 = self.upconv(x1)
    x3 = self.scaled(x2)
    return x3