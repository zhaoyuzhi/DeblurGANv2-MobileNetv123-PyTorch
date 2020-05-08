import os
import cv2
import skimage
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

import networks

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the network
    fpn_load_name, generator = networks.get_generator(opt)
    if opt.pre_train:
        # Init the network
        generator = networks.init_generator(generator, opt)
        print('Initialize network with %s type' % opt.init_type)
        pretrained_net = torch.load(fpn_load_name)
        load_dict(generator.backbone, pretrained_net)
        print('Generator is created!')
    else:
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name)
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator

def create_discriminator(opt):
    # Initialize the network
    discriminator = networks.get_discriminator(opt)
    # Init the network
    networks.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize network with %s type' % opt.init_type)
    print('Discriminators is created!')
    return discriminator
    
def create_perceptualnet(opt):
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = networks.get_perceptualnet()
    # Pre-trained VGG-16
    pretrained_dict = torch.load(opt.perceptualnet_name)
    load_dict(perceptualnet, pretrained_dict)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 128.0 + 128.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_sample_png_test(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 128.0 + 128.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret
    
def get_last_2paths(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.png':
                wholepath = os.path.join(root, filespath)
                last_2paths = os.path.join(wholepath.split('/')[-2], wholepath.split('/')[-1])
                ret.append(last_2paths)
    return ret
    
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

'''
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GradLoss(nn.Module):
    def __init__(self, GradLoss_weight = 1):
        super(GradLoss, self).__init__()
        self.GradLoss_weight = GradLoss_weight
        self.MSEloss = nn.MSELoss()

    def forward(self, x, y):
        h_x = x.size()[2]
        w_x = x.size()[3]

        x_h_grad = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        x_w_grad = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        y_h_grad = y[:, :, 1:, :] - y[:, :, :h_x - 1, :]
        y_w_grad = y[:, :, :, 1:] - y[:, :, :, :w_x - 1]
        
        h_loss = self.MSEloss(x_h_grad, y_h_grad)
        w_loss = self.MSEloss(x_w_grad, y_w_grad)
        
        return self.GradLoss_weight * (h_loss + w_loss)
'''

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # Initialization parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train ot not')
    parser.add_argument('--fpn_load_name', type = str, default = './trained_models/mobilenetv3_small_rgb_epoch150_bs256.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--network_type', type = str, default = 'MobileNetv3_small_DSC', help = 'pad type of networks')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--filter_channels', type = int, default = 128, help = 'start channels for generator')
    parser.add_argument('--mid_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for discriminator')
    parser.add_argument('--nlayer_d', type = int, default = 3, help = 'number of downsampling layers of discriminator')
    parser.add_argument('--use_sigmoid', type = bool, default = False, help = 'whether to use sigmoid layer at the end of discriminator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    opt = parser.parse_args()

    net = create_generator(opt).cuda()
    #torch.save(net.state_dict(), 'test.pth')
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    print(b.shape)
    '''
    net = create_discriminator(opt).cuda()
    #torch.save(net.state_dict(), 'test.pth')
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    print(b.shape)
    '''
    