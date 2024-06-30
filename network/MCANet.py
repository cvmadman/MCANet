from audioop import bias
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

class CBAM_Module(nn.Module):
    def __init__(self, dim, in_channels, ratio, kernel_size):
        super(CBAM_Module, self).__init__()
        self.avg_pool = getattr(nn, "AdaptiveAvgPool{0}d".format(dim))(1) #(1)压缩尺寸为1*1*1
        self.max_pool = getattr(nn, "AdaptiveMaxPool{0}d".format(dim))(1)
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.fc1 = conv_fn(in_channels, in_channels // ratio, kernel_size=1, padding=0,bias=False)
        self.relu = nn.ReLU()
        self.fc2 = conv_fn(in_channels // ratio, in_channels, kernel_size=1, padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = conv_fn(2, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # Channel attention module:（Mc(f) = σ(MLP(AvgPool(f)) + MLP(MaxPool(f)))）
        module_input = x
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg * mx)
        x = module_input * x
        # Spatial attention module:Ms (f) = σ( f7×7( AvgPool(f) ; MaxPool(F)] )))
        # module_input = x
        # avg = torch.mean(x, dim=1, keepdim=True)
        # mx, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat((avg, mx), dim=1)
        # x = self.sigmoid(self.conv(x))
        # x = module_input * x
        return x

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


    
class UNet(nn.Module):
    def __init__(self,vol_size):
        super(UNet, self).__init__()
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
        x = self.full(x)                                   #16
        E1 = self.full13(self.full12(self.full11(self.e1(x))))                         #16
        E2 = self.full23(self.full22(self.full21(self.e2(E1))))                        #32
        # E3 = self.full33(self.full32(self.full31(self.e3(E2))))                        #64
        E3 = self.full31(self.e3(E2))
        E4 = self.full4_1(self.e4(E3))                      #64 64
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

class UNet1(nn.Module):
    def __init__(self,vol_size):
        super(UNet1, self).__init__()
        self.full = Conv_Block(2, 16)
        self.full2 = Dilated_Conv_Block(16, 16, 2, 2)
        self.full3 = Dilated_Conv_Block(16, 16, 3, 3)    #Dilated_Conv_Block(2, 16, 1, 1)
        self.e1 = DownSample(16, 16)
        self.full11 = Conv_Block(16, 16)
        self.full22 = Dilated_Conv_Block(32, 32, 2, 2)
        self.full23 = Dilated_Conv_Block(32, 32, 3, 3)
        self.e2 = DownSample(16, 32)
        self.full21 = Conv_Block(32, 32)
        self.e3 = DownSample(32, 64)
        self.full31 = Conv_Block(64, 64)
        self.e4 = DownSample(64, 64)
        self.full41 = Conv_Block(64, 64)
        self.Agate1 = AG(in_channels=64,gating_channels=64,inter_channels=None)
        # self.upsample1 = nn.ConvTranspose3d(64, 32, 4,stride=2,padding=1)
        self.full5 = Conv_Block(96, 64)
        self.Agate2 = AG(in_channels=32,gating_channels=64,inter_channels=None)
        # self.upsample2 = nn.ConvTranspose3d(64,32,4,stride=2,padding=1)
        self.full6 = Conv_Block(64, 32)
        self.Agate3 = AG(in_channels=16,gating_channels=32,inter_channels=16)
        # self.upsample3 = nn.ConvTranspose3d(32,16,4,stride=2,padding=1)
        self.full7 = Conv_Block(32, 16)
        self.Agate4 = AG(in_channels=16,gating_channels=16,inter_channels=16)
        # self.upsample4 = nn.ConvTranspose3d(16,16,4,stride=2,padding=1)
        self.f1 = Conv_Block(32, 16)
        self.f2 = Conv_Block(16, 16)
        conv_fn = getattr(nn, 'Conv3d')
        self.flow = conv_fn(16,3, kernel_size=3, stride=1,padding=1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.full3(self.full2(self.full(x)))                             #16
        E1 = self.full23(self.full22(self.full11(self.e1(x))))               #16
        E2 = self.full21(self.e2(E1))                                        #32            
        E3 = self.full31(self.e3(E2))
        E4 = self.full41(self.e4(E3))                                        #64 
        A4 = self.Agate1(E3, E4)
        Up4 = F.interpolate(E4,scale_factor=2, mode='nearest')
        AU4 = torch.cat([Up4, A4], dim=1)
        D4 = self.full5(AU4)
        A3 = self.Agate2(E2, D4)
        Up3 = F.interpolate(D4,scale_factor=2, mode='nearest')
        AU3 = torch.cat([A3, Up3], dim=1)
        D3 = self.full6(AU3)
        A2 = self.Agate3(E1, D3)
        Up2 = F.interpolate(D3,scale_factor=2, mode='nearest')
        AU2 = torch.cat([A2, Up2], dim=1)
        D2 = self.full7(AU2)
        A1 = self.Agate4(x, D2)
        Up1 = F.interpolate(D2,scale_factor=2, mode='nearest')
        AU1 = torch.cat([A1, Up1],dim=1)
        F1 = self.f1(AU1)
        F2 = self.f2(F1)
        flow = self.flow(F2)
        return flow


class U_UNet(nn.Module):
    def __init__(self,vol_size):
        super(U_UNet, self).__init__()
        self.U_Net = UNet(vol_size)
        self.U_U_Net = UNet(vol_size)
        self.STN = SpatialTransformer(vol_size)
        conv_fn = getattr(nn, 'Conv3d')
        self.flow = conv_fn(6,3, kernel_size=3, stride=1,padding=1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
    def forward(self, src, tgt):
        flow1 = self.U_Net(src,tgt)
        warped1 = self.STN(src, flow1)
        flow2 = self.U_U_Net(warped1, tgt)
        f = torch.cat([flow1, flow2],dim=1)
        flow = self.flow(f)
        
        return flow
