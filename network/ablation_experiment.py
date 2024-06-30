from audioop import bias
from tkinter import E
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from Model.AGs import GridAttentionBlock3D as AG

class DownSample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DownSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x):
        return self.layers(x)

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,3,1,1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(out_channel)
        )
    def forward(self,x):
        return self.layers(x)

class Dilated_Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel,padding,dilate):
        super(Dilated_Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,3,1,padding,dilation=dilate),      #n=k+(k-1)*(dila-1)  [(1,1)(2,2),(3,3)]
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(out_channel)
        )
    def forward(self,x):
        return self.layers(x)

class Upsample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Upsample,self).__init__()
        self.layer = Conv_Block(in_channel,out_channel)
        # self.cbam = CBAM_Module(3,out_channel,16,7)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x):
        # up = F.interpolate(x,scale_factor=2, mode='nearest')
        out = self.layer(x)
        # weight = self.cbam(out)
        up = self.upsample(out)
        return up

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs,align_corners=True, mode=self.mode)

class U_woDC_Net(nn.Module):
    def __init__(self,vol_size):
        super(U_woDC_Net, self).__init__()
        self.full = Conv_Block(2, 16)    #Dilated_Conv_Block(2, 16, 1, 1)
        self.e1 = DownSample(16, 16)
        self.full11 = Dilated_Conv_Block(16, 16, 1, 1)
        # self.full12 = Dilated_Conv_Block(16, 16, 2, 2)
        # self.full13 = Dilated_Conv_Block(16, 16, 3, 3)
        self.e2 = DownSample(16, 32)
        self.full21 = Dilated_Conv_Block(32, 32, 1, 1)
        # self.full22 = Dilated_Conv_Block(32, 32, 2, 2)
        # self.full23 = Dilated_Conv_Block(32, 32, 3, 3)
        self.e3 = DownSample(32, 64)
        self.full31 = Conv_Block(64, 64)
        # self.full32 = Dilated_Conv_Block(64, 64, 2, 2)
        # self.full33 = Dilated_Conv_Block(64, 64, 3, 3)
        self.e4 = DownSample(64, 64)
        self.full4_1 = Dilated_Conv_Block(64, 64, 1, 1)
        self.Agate1 = AG(in_channels=64,gating_channels=64,inter_channels=None)
        self.upsample1 = nn.ConvTranspose3d(64, 32, 4,stride=2,padding=1)
        self.full5 = Conv_Block(96, 64)
        self.Agate2 = AG(in_channels=32,gating_channels=64,inter_channels=None)
        self.upsample2 = nn.ConvTranspose3d(64,32,4,stride=2,padding=1)
        self.full6 = Conv_Block(64, 32)
        self.Agate3 = AG(in_channels=16,gating_channels=32,inter_channels=16)
        self.upsample3 = nn.ConvTranspose3d(32,16,4,stride=2,padding=1)
        self.full7 = Conv_Block(32, 16)
        self.Agate4 = AG(in_channels=16,gating_channels=16,inter_channels=16)
        self.upsample4 = nn.ConvTranspose3d(16,16,4,stride=2,padding=1)
        self.f1 = Conv_Block(32, 16)
        self.f2 = Conv_Block(16, 16)
        conv_fn = getattr(nn, 'Conv3d')
        self.flow = conv_fn(16,3, kernel_size=3, stride=1,padding=1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))


    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.full(x)                                           #16
        E1 = self.full11(self.e1(x))                               #16
        E2 = self.full21(self.e2(E1))                              #32
        # E3 = self.full33(self.full32(self.full31(self.e3(E2))))  #64
        E3 = self.full31(self.e3(E2))
        E4 = self.full4_1(self.e4(E3))                             #64 
        # E5 = self.full4_2(E4)
        A4 = self.Agate1(E3, E4)
        Up4 = self.upsample1(E4)
        AU4 = torch.cat([Up4, A4], dim=1)
        D4 = self.full5(AU4)
        A3 = self.Agate2(E2, D4)
        Up3 = self.upsample2(D4)
        AU3 = torch.cat([A3, Up3], dim=1)
        D3 = self.full6(AU3)
        A2 = self.Agate3(E1, D3)
        Up2 = self.upsample3(D3)
        AU2 = torch.cat([A2, Up2], dim=1)
        D2 = self.full7(AU2)
        A1 = self.Agate4(x, D2)
        Up1 = self.upsample4(D2)
        AU1 = torch.cat([A1, Up1],dim=1)
        F1 = self.f1(AU1)
        F2 = self.f2(F1)
        flow = self.flow(F2)

        return flow

class U_woAG_Net(nn.Module):
    def __init__(self,vol_size):
        super(U_woAG_Net, self).__init__()
        self.full = Conv_Block(2, 16)    #Dilated_Conv_Block(2, 16, 1, 1)
        self.e1 = DownSample(16, 16)
        self.full11 = Dilated_Conv_Block(16, 16, 1, 1)
        self.full12 = Dilated_Conv_Block(16, 16, 2, 2)
        self.full13 = Dilated_Conv_Block(16, 16, 3, 3)
        self.e2 = DownSample(16, 32)
        self.full21 = Dilated_Conv_Block(32, 32, 1, 1)
        self.full22 = Dilated_Conv_Block(32, 32, 2, 2)
        self.full23 = Dilated_Conv_Block(32, 32, 3, 3)
        self.e3 = DownSample(32, 64)
        self.full31 = Conv_Block(64, 64)
        self.e4 = DownSample(64, 64)
        self.full4_1 = Dilated_Conv_Block(64, 64, 1, 1)
        # self.Agate1 = AG(in_channels=64,gating_channels=64,inter_channels=None)
        self.upsample1 = nn.ConvTranspose3d(64, 32, 4,stride=2,padding=1)
        self.full5 = Conv_Block(96, 64)
        # self.Agate2 = AG(in_channels=32,gating_channels=64,inter_channels=None)
        self.upsample2 = nn.ConvTranspose3d(64,32,4,stride=2,padding=1)
        self.full6 = Conv_Block(64, 32)
        # self.Agate3 = AG(in_channels=16,gating_channels=32,inter_channels=16)
        self.upsample3 = nn.ConvTranspose3d(32,16,4,stride=2,padding=1)
        self.full7 = Conv_Block(32, 16)
        # self.Agate4 = AG(in_channels=16,gating_channels=16,inter_channels=16)
        self.upsample4 = nn.ConvTranspose3d(16,16,4,stride=2,padding=1)
        self.f1 = Conv_Block(32, 16)
        self.f2 = Conv_Block(16, 16)
        conv_fn = getattr(nn, 'Conv3d')
        self.flow = conv_fn(16,3, kernel_size=3, stride=1,padding=1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))


    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.full(x)                                           #16
        E1 = self.full13(self.full12(self.full11(self.e1(x))))                               #16
        E2 = self.full23(self.full22(self.full21(self.e2(E1))))                              #32
        # E3 = self.full33(self.full32(self.full31(self.e3(E2))))  #64
        E3 = self.full31(self.e3(E2))
        E4 = self.full4_1(self.e4(E3))                             #64 
        # E5 = self.full4_2(E4)
        Up4 = self.upsample1(E4)
        AU4 = torch.cat([Up4, E3], dim=1)
        D4 = self.full5(AU4)
        Up3 = self.upsample2(D4)
        AU3 = torch.cat([E2, Up3], dim=1)
        D3 = self.full6(AU3)
        Up2 = self.upsample3(D3)
        AU2 = torch.cat([E1, Up2], dim=1)
        D2 = self.full7(AU2)
        Up1 = self.upsample4(D2)
        AU1 = torch.cat([x, Up1],dim=1)
        F1 = self.f1(AU1)
        F2 = self.f2(F1)
        flow = self.flow(F2)

        return flow

class U_woAGDC_Net(nn.Module):
    def __init__(self,vol_size):
        super(U_woAGDC_Net, self).__init__()
        self.full = Conv_Block(2, 16)    #Dilated_Conv_Block(2, 16, 1, 1)
        self.e1 = DownSample(16, 16)
        self.full11 = Dilated_Conv_Block(16, 16, 1, 1)
        self.e2 = DownSample(16, 32)
        self.full21 = Dilated_Conv_Block(32, 32, 1, 1)
        self.e3 = DownSample(32, 64)
        self.full31 = Conv_Block(64, 64)
        self.e4 = DownSample(64, 64)
        self.full4_1 = Dilated_Conv_Block(64, 64, 1, 1)
        # self.Agate1 = AG(in_channels=64,gating_channels=64,inter_channels=None)
        self.upsample1 = nn.ConvTranspose3d(64, 32, 4,stride=2,padding=1)
        self.full5 = Conv_Block(96, 64)
        # self.Agate2 = AG(in_channels=32,gating_channels=64,inter_channels=None)
        self.upsample2 = nn.ConvTranspose3d(64,32,4,stride=2,padding=1)
        self.full6 = Conv_Block(64, 32)
        # self.Agate3 = AG(in_channels=16,gating_channels=32,inter_channels=16)
        self.upsample3 = nn.ConvTranspose3d(32,16,4,stride=2,padding=1)
        self.full7 = Conv_Block(32, 16)
        # self.Agate4 = AG(in_channels=16,gating_channels=16,inter_channels=16)
        self.upsample4 = nn.ConvTranspose3d(16,16,4,stride=2,padding=1)
        self.f1 = Conv_Block(32, 16)
        self.f2 = Conv_Block(16, 16)
        conv_fn = getattr(nn, 'Conv3d')
        self.flow = conv_fn(16,3, kernel_size=3, stride=1,padding=1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))


    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.full(x)                                           #16
        E1 = self.full11(self.e1(x))                               #16
        E2 = self.full21(self.e2(E1))                              #32
        # E3 = self.full33(self.full32(self.full31(self.e3(E2))))  #64
        E3 = self.full31(self.e3(E2))
        E4 = self.full4_1(self.e4(E3))                             #64 
        # E5 = self.full4_2(E4)
        Up4 = self.upsample1(E4)
        AU4 = torch.cat([Up4, E3], dim=1)
        D4 = self.full5(AU4)
        Up3 = self.upsample2(D4)
        AU3 = torch.cat([E2, Up3], dim=1)
        D3 = self.full6(AU3)
        Up2 = self.upsample3(D3)
        AU2 = torch.cat([E1, Up2], dim=1)
        D2 = self.full7(AU2)
        Up1 = self.upsample4(D2)
        AU1 = torch.cat([x, Up1],dim=1)
        F1 = self.f1(AU1)
        F2 = self.f2(F1)
        flow = self.flow(F2)

        return flow
