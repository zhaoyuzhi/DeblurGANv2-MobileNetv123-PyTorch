import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *

# ----------------------------------------
#                Generator
# ----------------------------------------
class DeblurGANv2(nn.Module):
    def __init__(self, opt, fpn, n1, n2, n3, n4, n5):
        super(DeblurGANv2, self).__init__()
        # FPN backbone
        self.backbone = fpn
        # Feature Pyramid Network
        self.fpn1 = Conv2dLayer(n1, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn2 = Conv2dLayer(n2, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn3 = Conv2dLayer(n3, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn4 = Conv2dLayer(n4, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn5 = Conv2dLayer(n5, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpnup1 = Conv2dLayer(opt.filter_channels, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fpnup2 = Conv2dLayer(opt.filter_channels, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fpnup3 = Conv2dLayer(opt.filter_channels, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Final Conv
        self.head1 = nn.Sequential(
            Conv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            Conv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.head2 = nn.Sequential(
            Conv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            Conv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.head3 = nn.Sequential(
            Conv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            Conv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.head4 = nn.Sequential(
            Conv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            Conv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.fusion1 = Conv2dLayer(opt.mid_channels * 4, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fusion2 = Conv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fusion3 = Conv2dLayer(opt.mid_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'tanh', norm = 'none')
        
    def forward(self, x):

        residual = x
        
        # FPN backbone feature extraction
        x1, x2, x3, x4, x5 = self.backbone(x)
        # Process FPN
        x1 = self.fpn1(x1)                                                                  # out: batch * 128 * 128 * 128
        x2 = self.fpn2(x2)                                                                  # out: batch * 128 * 64 * 64
        x3 = self.fpn3(x3)                                                                  # out: batch * 128 * 32 * 32
        x4 = self.fpn4(x4)                                                                  # out: batch * 128 * 16 * 16
        x5 = self.fpn5(x5)                                                                  # out: batch * 128 * 8 * 8
        #map5 = x5                                                                          # out: batch * 128 * 8 * 8
        map4 = self.fpnup1(x4 + F.upsample(x5, scale_factor = 2, mode = 'nearest'))         # out: batch * 128 * 16 * 16
        map3 = self.fpnup2(x3 + F.upsample(map4, scale_factor = 2, mode = 'nearest'))       # out: batch * 128 * 32 * 32
        map2 = self.fpnup3(x2 + F.upsample(map3, scale_factor = 2, mode = 'nearest'))       # out: batch * 128 * 64 * 64
        #map1 = x1                                                                          # out: batch * 128 * 128 * 128
        # Final Upsample
        f5 = F.upsample(self.head1(x5), scale_factor = 8, mode = 'nearest')                 # out: batch * 64 * 64 * 64
        f4 = F.upsample(self.head2(map4), scale_factor = 4, mode = 'nearest')               # out: batch * 64 * 64 * 64
        f3 = F.upsample(self.head3(map3), scale_factor = 2, mode = 'nearest')               # out: batch * 64 * 64 * 64
        f2 = self.head4(map2)                                                               # out: batch * 64 * 64 * 64
        fusion = torch.cat((f2, f3, f4, f5), 1)                                             # out: batch * 256 * 64 * 64
        fusion = F.upsample(self.fusion1(fusion), scale_factor = 2, mode = 'nearest')       # out: batch * 128 * 128 * 128
        fusion = F.upsample(self.fusion2(fusion + x1), scale_factor = 2, mode = 'nearest')  # out: batch * 64 * 256 * 256
        fusion = self.fusion3(fusion)                                                       # out: batch * 3 * 256 * 256

        out = residual - fusion
        out = torch.clamp(out, min = -1, max = 1)

        return out

class DeblurGANv2_DSC(nn.Module):
    def __init__(self, opt, fpn, n1, n2, n3, n4, n5):
        super(DeblurGANv2_DSC, self).__init__()
        # FPN backbone
        self.backbone = fpn
        # Feature Pyramid Network
        self.fpn1 = Conv2dLayer(n1, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn2 = Conv2dLayer(n2, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn3 = Conv2dLayer(n3, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn4 = Conv2dLayer(n4, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpn5 = Conv2dLayer(n5, opt.filter_channels, 1, 1, 0, pad_type = opt.pad, activation = 'none', norm = 'none')
        self.fpnup1 = DWConv2dLayer(opt.filter_channels, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fpnup2 = DWConv2dLayer(opt.filter_channels, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fpnup3 = DWConv2dLayer(opt.filter_channels, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Final Conv
        self.head1 = nn.Sequential(
            DWConv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            DWConv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.head2 = nn.Sequential(
            DWConv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            DWConv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.head3 = nn.Sequential(
            DWConv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            DWConv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.head4 = nn.Sequential(
            DWConv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none'),
            DWConv2dLayer(opt.mid_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        )
        self.fusion1 = DWConv2dLayer(opt.mid_channels * 4, opt.filter_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fusion2 = DWConv2dLayer(opt.filter_channels, opt.mid_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.fusion3 = DWConv2dLayer(opt.mid_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'tanh', norm = 'none')
        
    def forward(self, x):

        residual = x
        
        # FPN backbone feature extraction
        x1, x2, x3, x4, x5 = self.backbone(x)
        # Process FPN
        x1 = self.fpn1(x1)                                                                  # out: batch * 128 * 128 * 128
        x2 = self.fpn2(x2)                                                                  # out: batch * 128 * 64 * 64
        x3 = self.fpn3(x3)                                                                  # out: batch * 128 * 32 * 32
        x4 = self.fpn4(x4)                                                                  # out: batch * 128 * 16 * 16
        x5 = self.fpn5(x5)                                                                  # out: batch * 128 * 8 * 8
        #map5 = x5                                                                          # out: batch * 128 * 8 * 8
        map4 = self.fpnup1(x4 + F.upsample(x5, scale_factor = 2, mode = 'nearest'))         # out: batch * 128 * 16 * 16
        map3 = self.fpnup2(x3 + F.upsample(map4, scale_factor = 2, mode = 'nearest'))       # out: batch * 128 * 32 * 32
        map2 = self.fpnup3(x2 + F.upsample(map3, scale_factor = 2, mode = 'nearest'))       # out: batch * 128 * 64 * 64
        #map1 = x1                                                                          # out: batch * 128 * 128 * 128
        # Final Upsample
        f5 = F.upsample(self.head1(x5), scale_factor = 8, mode = 'nearest')                 # out: batch * 64 * 64 * 64
        f4 = F.upsample(self.head2(map4), scale_factor = 4, mode = 'nearest')               # out: batch * 64 * 64 * 64
        f3 = F.upsample(self.head3(map3), scale_factor = 2, mode = 'nearest')               # out: batch * 64 * 64 * 64
        f2 = self.head4(map2)                                                               # out: batch * 64 * 64 * 64
        fusion = torch.cat((f2, f3, f4, f5), 1)                                             # out: batch * 256 * 64 * 64
        fusion = F.upsample(self.fusion1(fusion), scale_factor = 2, mode = 'nearest')       # out: batch * 128 * 128 * 128
        fusion = F.upsample(self.fusion2(fusion + x1), scale_factor = 2, mode = 'nearest')  # out: batch * 64 * 256 * 256
        fusion = self.fusion3(fusion)                                                       # out: batch * 3 * 256 * 256

        out = residual - fusion
        out = torch.clamp(out, min = -1, max = 1)

        return out
        
# ----------------------------------------
#               Discriminator
# ----------------------------------------
# This is a kind of PatchGAN. Patch is implied in the output.
class NlayerPatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(NlayerPatchDiscriminator, self).__init__()
        # Down sampling
        nlayerlist = []
        nlayerlist.append(Conv2dLayer(opt.out_channels, opt.start_channels, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True))
        for i in range(1, opt.nlayer_d):
            in_channels = opt.start_channels * (2 ** (i - 1))
            out_channels = min(opt.start_channels * (2 ** i), 512)    # maximum channel number equals to 512
            nlayerlist.append(Conv2dLayer(in_channels, out_channels, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True))
        nlayerlist.append(Conv2dLayer(out_channels, out_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True))
        nlayerlist.append(Conv2dLayer(out_channels, 1, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True))
        if opt.use_sigmoid:
            nlayerlist.append(nn.Sigmoid())
        self.conv = nn.Sequential(*nlayerlist)

    def forward(self, x):
        x = self.conv(x)
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
