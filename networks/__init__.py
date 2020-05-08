import torch.nn.init as init

from networks.network import *
from networks.network_ResNet import *
from networks.network_MobileNetv1 import *
from networks.network_MobileNetv2 import *
from networks.network_MobileNetv3 import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)
    
def get_generator(opt):
    if opt.network_type == 'ResNet50_BN':
        fpn_load_name = './trained_models/resnet50_bn_rgb_epoch150_bs256.pth'
        fpn = ResNet_BN_FPN(Bottleneck_BN, [3, 4, 3, 3])
        n1, n2, n3, n4, n5 = 64, 256, 512, 1024, 2048
        deblurganv2 = DeblurGANv2(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'ResNet50_IN':
        fpn_load_name = './trained_models/resnet50_in_rgb_epoch150_bs256.pth'
        fpn = ResNet_IN_FPN(Bottleneck_IN, [3, 4, 3, 3])
        n1, n2, n3, n4, n5 = 64, 256, 512, 1024, 2048
        deblurganv2 = DeblurGANv2(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv1':
        fpn_load_name = './trained_models/mobilenetv1_rgb_epoch150_bs256.pth'
        fpn = MobileNetV1_FPN()
        n1, n2, n3, n4, n5 = 64, 128, 256, 512, 1024
        deblurganv2 = DeblurGANv2(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv2':
        fpn_load_name = './trained_models/mobilenetv2_rgb_epoch150_bs256.pth'
        fpn = MobileNetV2_FPN()
        n1, n2, n3, n4, n5 = 16, 24, 32, 96, 320
        deblurganv2 = DeblurGANv2(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv3_large':
        fpn_load_name = './trained_models/mobilenetv3_large_rgb_epoch150_bs256.pth'
        fpn = MobileNetV3_large_FPN()
        n1, n2, n3, n4, n5 = 16, 24, 40, 112, 160
        deblurganv2 = DeblurGANv2(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv3_small':
        fpn_load_name = './trained_models/mobilenetv3_small_rgb_epoch150_bs256.pth'
        fpn = MobileNetV3_small_FPN()
        n1, n2, n3, n4, n5 = 16, 16, 24, 48, 96
        deblurganv2 = DeblurGANv2(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'ResNet50_BN_DSC':
        fpn_load_name = './trained_models/resnet50_bn_rgb_epoch150_bs256.pth'
        fpn = ResNet_BN_FPN(Bottleneck_BN, [3, 4, 3, 3])
        n1, n2, n3, n4, n5 = 64, 256, 512, 1024, 2048
        deblurganv2 = DeblurGANv2_DSC(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'ResNet50_IN_DSC':
        fpn_load_name = './trained_models/resnet50_in_rgb_epoch150_bs256.pth'
        fpn = ResNet_IN_FPN(Bottleneck_IN, [3, 4, 3, 3])
        n1, n2, n3, n4, n5 = 64, 256, 512, 1024, 2048
        deblurganv2 = DeblurGANv2_DSC(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv1_DSC':
        fpn_load_name = './trained_models/mobilenetv1_rgb_epoch150_bs256.pth'
        fpn = MobileNetV1_FPN()
        n1, n2, n3, n4, n5 = 64, 128, 256, 512, 1024
        deblurganv2 = DeblurGANv2_DSC(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv2_DSC':
        fpn_load_name = './trained_models/mobilenetv2_rgb_epoch150_bs256.pth'
        fpn = MobileNetV2_FPN()
        n1, n2, n3, n4, n5 = 16, 24, 32, 96, 320
        deblurganv2 = DeblurGANv2_DSC(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv3_large_DSC':
        fpn_load_name = './trained_models/mobilenetv3_large_rgb_epoch150_bs256.pth'
        fpn = MobileNetV3_large_FPN()
        n1, n2, n3, n4, n5 = 16, 24, 40, 112, 160
        deblurganv2 = DeblurGANv2_DSC(opt, fpn, n1, n2, n3, n4, n5)
    if opt.network_type == 'MobileNetv3_small_DSC':
        fpn_load_name = './trained_models/mobilenetv3_small_rgb_epoch150_bs256.pth'
        fpn = MobileNetV3_small_FPN()
        n1, n2, n3, n4, n5 = 16, 16, 24, 48, 96
        deblurganv2 = DeblurGANv2_DSC(opt, fpn, n1, n2, n3, n4, n5)
    return fpn_load_name, deblurganv2

def get_discriminator(opt):
    discriminator = NlayerPatchDiscriminator(opt)
    return discriminator

def get_perceptualnet():
    perceptualnet = PerceptualNet()
    return perceptualnet

def init_generator(generator, opt):
    weights_init(generator.fpn1, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpn2, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpn3, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpn4, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpn5, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpnup1, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpnup2, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fpnup3, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.head1, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.head2, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.head3, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.head4, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fusion1, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fusion2, init_type = opt.init_type, init_gain = opt.init_gain)
    weights_init(generator.fusion3, init_type = opt.init_type, init_gain = opt.init_gain)
    return generator
