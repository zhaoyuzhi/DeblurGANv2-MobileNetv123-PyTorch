import torch
import torch.nn as nn

###========================== MobileNetv2 framework ==========================
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_channels = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # dw: depth-wise convolution
        # pw: point-wise convolution
        # pw-linear: point-wise convolution without activation
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, hidden_channels, 3, stride, 1, groups = hidden_channels, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # pw-linear
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # dw
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups = hidden_channels, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # pw-linear
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

'''
interverted_residual_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]
'''

class MobileNetV2(nn.Module):
    def __init__(self, last_channels = 1280, n_class = 1000):
        super(MobileNetV2, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace = True)
        )
        # InvertedResidual blocks
        self.conv2 = InvertedResidual(32, 16, 1, 1)
        self.conv3 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
        )
        self.conv4 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
        )
        self.conv5 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
        )
        self.conv6 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
        )
        self.conv7 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)
        )
        self.conv8 = InvertedResidual(160, 320, 1, 6)
        # Last Conv
        self.conv9 = nn.Sequential(
            nn.Conv2d(320, last_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace = True)
        )
        # Classifier
        self.classifier = nn.Linear(last_channels, n_class)

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)                                   # out: B * 32 * 112 * 112
        x = self.conv2(x)                                   # out: B * 16 * 112 * 112
        x = self.conv3(x)                                   # out: B * 24 * 56 * 56
        x = self.conv4(x)                                   # out: B * 32 * 28 * 28
        x = self.conv5(x)                                   # out: B * 64 * 14 * 14
        x = self.conv6(x)                                   # out: B * 96 * 14 * 14
        x = self.conv7(x)                                   # out: B * 160 * 7 * 7
        x = self.conv8(x)                                   # out: B * 320 * 7 * 7
        x = self.conv9(x)                                   # out: B * 1280 * 7 * 7
        # classifier
        x = x.mean(3).mean(2)                               # out: B * 1280 (global avg pooling)
        x = self.classifier(x)                              # out: B * 1000
        return x

class MobileNetV2_FPN(nn.Module):
    def __init__(self):
        super(MobileNetV2_FPN, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace = True)
        )
        # InvertedResidual blocks
        self.conv2 = InvertedResidual(32, 16, 1, 1)
        self.conv3 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
        )
        self.conv4 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
        )
        self.conv5 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
        )
        self.conv6 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
        )
        self.conv7 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)
        )
        self.conv8 = InvertedResidual(160, 320, 1, 6)

    def forward(self, x):
        # feature extraction
        x1 = self.conv1(x)                                  # out: B * 32 * 112 * 112
        x1 = self.conv2(x1)                                 # out: B * 16 * 112 * 112
        x2 = self.conv3(x1)                                 # out: B * 24 * 56 * 56
        x3 = self.conv4(x2)                                 # out: B * 32 * 28 * 28
        x4 = self.conv5(x3)                                 # out: B * 64 * 14 * 14
        x4 = self.conv6(x4)                                 # out: B * 96 * 14 * 14
        x5 = self.conv7(x4)                                 # out: B * 160 * 7 * 7
        x5 = self.conv8(x5)                                 # out: B * 320 * 7 * 7
        return x1, x2, x3, x4, x5

if __name__ == "__main__":
    net = MobileNetV2()
    a = torch.randn(1, 3, 224, 224)
    b = net(a)
    print(b.shape)
