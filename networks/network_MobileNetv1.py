import torch
import torch.nn as nn

###========================== MobileNetv1 framework ==========================
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            # pw
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)

class MobileNetV1(nn.Module):
    def __init__(self, n_class = 1000):
        super(MobileNetV1, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )
        # DWConv blocks
        self.conv2 = DWConv(32, 64, 1)
        self.conv3 = DWConv(64, 128, 2)
        self.conv4 = DWConv(128, 128, 1)
        self.conv5 = DWConv(128, 256, 2)
        self.conv6 = DWConv(256, 256, 1)
        self.conv7 = DWConv(256, 512, 2)
        self.conv8 = DWConv(512, 512, 1)
        self.conv9 = DWConv(512, 512, 1)
        self.conv10 = DWConv(512, 512, 1)
        self.conv11 = DWConv(512, 512, 1)
        self.conv12 = DWConv(512, 512, 1)
        self.conv13 = DWConv(512, 1024, 2)
        self.conv14 = DWConv(1024, 1024, 1)
        # Classifier
        self.classifier = nn.Linear(1024, n_class)

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)                                   # out: B * 32 * 112 * 112
        x = self.conv2(x)                                   # out: B * 64 * 112 * 112
        x = self.conv3(x)                                   # out: B * 128 * 56 * 56
        x = self.conv4(x)                                   # out: B * 128 * 56 * 56
        x = self.conv5(x)                                   # out: B * 256 * 28 * 28
        x = self.conv6(x)                                   # out: B * 256 * 28 * 28
        x = self.conv7(x)                                   # out: B * 512 * 14 * 14
        x = self.conv8(x)                                   # out: B * 512 * 14 * 14
        x = self.conv9(x)                                   # out: B * 512 * 14 * 14
        x = self.conv10(x)                                  # out: B * 512 * 14 * 14
        x = self.conv11(x)                                  # out: B * 512 * 14 * 14
        x = self.conv12(x)                                  # out: B * 512 * 14 * 14
        x = self.conv13(x)                                  # out: B * 1024 * 7 * 7
        x = self.conv14(x)                                  # out: B * 1024 * 7 * 7
        # classifier
        x = x.mean(3).mean(2)                               # out: B * 1024 (global avg pooling)
        x = self.classifier(x)                              # out: B * 1000
        return x

class MobileNetV1_FPN(nn.Module):
    def __init__(self):
        super(MobileNetV1_FPN, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )
        # DWConv blocks
        self.conv2 = DWConv(32, 64, 1)
        self.conv3 = DWConv(64, 128, 2)
        self.conv4 = DWConv(128, 128, 1)
        self.conv5 = DWConv(128, 256, 2)
        self.conv6 = DWConv(256, 256, 1)
        self.conv7 = DWConv(256, 512, 2)
        self.conv8 = DWConv(512, 512, 1)
        self.conv9 = DWConv(512, 512, 1)
        self.conv10 = DWConv(512, 512, 1)
        self.conv11 = DWConv(512, 512, 1)
        self.conv12 = DWConv(512, 512, 1)
        self.conv13 = DWConv(512, 1024, 2)
        self.conv14 = DWConv(1024, 1024, 1)

    def forward(self, x):
        # feature extraction
        x1 = self.conv1(x)                                  # out: B * 32 * 112 * 112
        x1 = self.conv2(x1)                                 # out: B * 64 * 112 * 112
        x2 = self.conv3(x1)                                 # out: B * 128 * 56 * 56
        x2 = self.conv4(x2)                                 # out: B * 128 * 56 * 56
        x3 = self.conv5(x2)                                 # out: B * 256 * 28 * 28
        x3 = self.conv6(x3)                                 # out: B * 256 * 28 * 28
        x4 = self.conv7(x3)                                 # out: B * 512 * 14 * 14
        x4 = self.conv8(x4)                                 # out: B * 512 * 14 * 14
        x4 = self.conv9(x4)                                 # out: B * 512 * 14 * 14
        x4 = self.conv10(x4)                                # out: B * 512 * 14 * 14
        x4 = self.conv11(x4)                                # out: B * 512 * 14 * 14
        x4 = self.conv12(x4)                                # out: B * 512 * 14 * 14
        x5 = self.conv13(x4)                                # out: B * 1024 * 7 * 7
        x5 = self.conv14(x5)                                # out: B * 1024 * 7 * 7
        return x1, x2, x3, x4, x5

if __name__ == "__main__":
    net = MobileNetV1()
    a = torch.randn(1, 3, 224, 224)
    b = net(a)
    print(b.shape)
