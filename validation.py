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
    parser.add_argument('--pre_train', type = bool, default = False, help = 'pre-train ot not')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--save_path', type = str, default = '', help = 'the save path to generated images')
    # Initialization parameters
    parser.add_argument('--network_type', type = str, default = 'MobileNetv3_small', help = 'pad type of networks')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--filter_channels', type = int, default = 128, help = 'start channels for generator')
    parser.add_argument('--mid_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for discriminator')
    # Dataset parameters
    parser.add_argument('--baseroot_test_blur', type = str, \
        default = 'E:\\dataset, task related\\Deblurring Dataset\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\test\\blur', \
            help = 'blurry image baseroot')
    parser.add_argument('--baseroot_test_sharp', type = str, \
        default = 'E:\\dataset, task related\\Deblurring Dataset\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\test\\sharp', \
            help = 'clean image baseroot')
    parser.add_argument('--crop_size', type = int, default = 0, help = 'crop size for each image')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = dataset.DeblurDataset_val(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.save_path)

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
        save_img_path = os.path.join(opt.save_path, imgname)
        