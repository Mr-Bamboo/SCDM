import os
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision
import tifffile
import random
import copy


class HyperDataset(data_utils.Dataset):
    def __init__(self, root_path, data_file, input_channel=3, output_channel=12, cropsize=128, gt_exist=True):
        self.data_files = np.loadtxt(data_file, dtype=np.str_)
        self.root_path = root_path
        self.cropsize = cropsize
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()
            ]
        )
        # gt_exist == True means stage 1
        self.gt_exist = gt_exist

    def __getitem__(self, item):
        in_ch = self.input_channel
        ou_ch = self.output_channel
        data_file = self.data_files[item]
        img_file = data_file
        img_file = os.path.join(self.root_path, img_file)
        img = tifffile.imread(img_file)
        hs, ws, h, w = self.randomcrop(img, self.cropsize)
        if self.gt_exist:
            if in_ch == 3:
                all = copy.deepcopy(img)
                all = all[hs:hs + h, ws:ws + w, 0:48]
                img = img[hs:hs + h, ws:ws + w, 0:48][:,:,::4]
            else:
                all = copy.deepcopy(img)
                all = all[hs:hs + h, ws:ws + w, 0:48]
                img = img[hs:hs + h, ws:ws + w, 0:48]
            img = torch.tensor((img.astype('float32') - 3693) / 7410)
            all = torch.tensor((all.astype('float32') - 3693) / 7410)
            img = img.permute(2, 0, 1)
            all = all.permute(2, 0, 1)
            if in_ch == 3:
                rgb = all[[4, 11, 22], :, :]
            else:
                rgb = img[::4,:,:]
        else:
            all = copy.deepcopy(img)
            all = all[hs:hs + h, ws:ws + w, :]
            img = img[hs:hs + h, ws:ws + w, :]
            img = torch.tensor((img.astype('float32') - 3693) / 7410)
            all = torch.tensor((all.astype('float32') - 3693) / 7410)
            img = img.permute(2, 0, 1)
            all = all.permute(2, 0, 1)
            rgb = img

        return img, rgb, all

    def __len__(self):
        return len(self.data_files)

    def randomcrop(self, img, crop_size):
        h, w, c = img.shape
        th = tw = crop_size
        if w == tw and h == th:
            return 0, 0, h, w
        if w < tw or h < th:
            print('the size of the image is not enough for crop!')
            exit(-1)
        i = random.randint(0, h - th - 10)
        j = random.randint(0, w - tw - 10)
        return i, j, th, tw
