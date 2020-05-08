import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.lateral4 = nn.Conv2d(160, 128, 1, bias=False)
        self.lateral3 = nn.Conv2d(64, 128, 1, bias=False)
        self.lateral2 = nn.Conv2d(32, 128, 1, bias=False)
        self.lateral1 = nn.Conv2d(24, 128, 1, bias=False)
        self.lateral0 = nn.Conv2d(16, 128//2, 1, bias=False)

        self.td1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.td2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.td3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
            
    def forward(self, x):
        enc0 = torch.randn(1, 16, 128, 128)
        enc1 = torch.randn(1, 24, 64, 64)
        enc2 = torch.randn(1, 32, 32, 32)
        enc3 = torch.randn(1, 64, 16, 16)
        enc4 = torch.randn(1, 160, 8, 8)

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)
        print(lateral1.shape, lateral2.shape, lateral3.shape, lateral4.shape, lateral0.shape)

        map4 = lateral4
        map3 = self.td3(lateral3+F.upsample(map4,scale_factor=2))
        map2 = self.td2(lateral2+F.upsample(map3,scale_factor=2))
        map1 = self.td1(lateral1+F.upsample(map2,scale_factor=2))
        print(map1.shape, map2.shape, map3.shape, map4.shape, lateral0.shape)
        return lateral0, map1, map2, map3, map4

net = FPN()
a = torch.randn(1, 3, 256, 256)
b = net(a)