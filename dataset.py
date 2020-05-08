import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

import utils

class DeblurDataset(Dataset):
    def __init__(self, opt, tag):
        self.opt = opt
        self.blur_imglist = []
        self.sharp_imglist = []
        if tag == 'train':
            imglist = utils.get_last_2paths(opt.baseroot_train_blur)
            for i in range(len(imglist)):
                self.blur_imglist.append(os.path.join(self.opt.baseroot_train_blur, imglist[i]))
                self.sharp_imglist.append(os.path.join(self.opt.baseroot_train_sharp, imglist[i]))
        if tag == 'val':
            imglist = utils.get_last_2paths(opt.baseroot_val_blur)
            for i in range(len(imglist)):
                self.blur_imglist.append(os.path.join(self.opt.baseroot_val_blur, imglist[i]))
                self.sharp_imglist.append(os.path.join(self.opt.baseroot_val_sharp, imglist[i]))

    def __getitem__(self, index):
        # Path of one image
        blur_img_path = self.blur_imglist[index]
        clean_img_path = self.sharp_imglist[index]

        # Read the images
        blur_img = cv2.imread(blur_img_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.imread(clean_img_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

        # Random cropping
        if self.opt.crop_size > 0:
            h, w = blur_img.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            blur_img = blur_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
            clean_img = clean_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]

        # Normalized to [-1, 1]
        blur_img = np.ascontiguousarray(blur_img, dtype = np.float32)
        blur_img = (blur_img - 128.0) / 128.0
        clean_img = np.ascontiguousarray(clean_img, dtype = np.float32)
        clean_img = (clean_img - 128.0) / 128.0

        # To PyTorch Tensor
        blur_img = torch.from_numpy(blur_img).permute(2, 0, 1).contiguous()
        clean_img = torch.from_numpy(clean_img).permute(2, 0, 1).contiguous()

        return blur_img, clean_img
    
    def __len__(self):
        return len(self.blur_imglist)

class DeblurDataset_val(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot_test_blur)

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        blur_img_path = os.path.join(self.opt.baseroot_test_blur, imgname)
        clean_img_path = os.path.join(self.opt.baseroot_test_sharp, imgname)

        # Read the images
        blur_img = cv2.imread(blur_img_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.imread(clean_img_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

        # Random cropping
        if self.opt.crop_size > 0:
            h, w = blur_img.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            blur_img = blur_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
            clean_img = clean_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]

        # Normalized to [-1, 1]
        blur_img = np.ascontiguousarray(blur_img, dtype = np.float32)
        blur_img = (blur_img - 128.0) / 128.0
        clean_img = np.ascontiguousarray(clean_img, dtype = np.float32)
        clean_img = (clean_img - 128.0) / 128.0

        # To PyTorch Tensor
        blur_img = torch.from_numpy(blur_img).permute(2, 0, 1).contiguous()
        clean_img = torch.from_numpy(clean_img).permute(2, 0, 1).contiguous()

        return blur_img, clean_img, imgname
    
    def __len__(self):
        return len(self.imglist)
