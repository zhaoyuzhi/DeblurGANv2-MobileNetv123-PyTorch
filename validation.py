import argparse
import os
import torch
import numpy as np
import cv2

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = False, help = 'pre_train or not')
    parser.add_argument('--load_name', type = str, default = './track1/G_epoch8000_bs8.pth', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './validation', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'track1', help = 'task name for loading networks, saving, and log')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    # Dataset parameters
    parser.add_argument('--baseroot_A', type = str, default = '', help = 'blurry image baseroot')
    parser.add_argument('--baseroot_B', type = str, default = '', help = 'clean image baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size for each image')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = dataset.DeblurDataset_val(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = os.path.join(opt.val_path, opt.task_name)
    utils.check_path(sample_folder)

    # forward
    for i, (true_input, true_target, imgname) in enumerate(test_loader):

        # To device
        true_input = true_input.cuda()
        true_target = true_target.cuda()

        # Forward propagation
        with torch.no_grad():
            fake_target = generator(true_input)

        # Save
        fake_target = fake_target.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        fake_target = cv2.cvtColor(fake_target, cv2.COLOR_BGR2RGB).astype(np.uint8)
        save_img_path = os.path.join(sample_folder, imgname)
        