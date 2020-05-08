import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train ot not')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'gopro', help = 'task name for loading networks, saving, and log')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--perceptualnet_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 300, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 150, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--patch_size', type = int, default = 64, help = 'the patch size for patch discriminator')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_l1', type = float, default = 0.5, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_percep', type = float, default = 0.006, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.01, help = 'coefficient for GAN Loss')
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
    parser.add_argument('--nlayer_d', type = int, default = 3, help = 'number of downsampling layers of discriminator')
    parser.add_argument('--use_sigmoid', type = bool, default = False, help = 'whether to use sigmoid layer at the end of discriminator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot_train_blur', type = str, \
        default = 'E:\\dataset, task related\\Deblurring Dataset\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\train\\blur', \
            help = 'blurry image baseroot')
    parser.add_argument('--baseroot_train_sharp', type = str, \
        default = 'E:\\dataset, task related\\Deblurring Dataset\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\train\\sharp', \
            help = 'clean image baseroot')
    parser.add_argument('--baseroot_val_blur', type = str, \
        default = 'E:\\dataset, task related\\Deblurring Dataset\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\test\\blur', \
            help = 'blurry image baseroot')
    parser.add_argument('--baseroot_val_sharp', type = str, \
        default = 'E:\\dataset, task related\\Deblurring Dataset\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\test\\sharp', \
            help = 'clean image baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size for each image')
    opt = parser.parse_args()
    print(opt)

    '''
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    '''
    
    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    if opt.pre_train:
        trainer.Pre_train(opt)
    else:
        trainer.Continue_train_WGAN(opt)
    