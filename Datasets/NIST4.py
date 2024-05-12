import sys

import imageio
sys.path.append('..')
from utils import mnt_reader_txt as mnt_reader
from torch.utils.data import Dataset
import torch
import logging
import imageio
from tqdm import tqdm
from glob import glob
from os import path
import numpy as np
import json

def T(img):
    return torch.from_numpy(np.expand_dims(img / 255., 0))

def imread(p, **args):
    return np.array(imageio.imread(p, pilmode='L'), dtype=np.uint8)


class NIST4(Dataset):

    def __init__(self, base_path, transform=T, ori_dir='L_ori_manual', mnt_dir='L_M', target_folder='/data4/albert/NISTSD27zx14/imgs/L'):

        self.mnt_dir = path.join(path.split(base_path)[0], mnt_dir)
        # print(self.ori_dir)
        self.base_path = base_path
        self.transform = transform
        self.target_folder = target_folder
        self.imgs, self.mnts, self.path, self.names = self.get_all_images()

    def get_all_images(self):
        all_p = glob(f'{self.base_path}/*')
        names, imgs, enhance, mnts, pathes = [], [], [], [], []
        for i, p in enumerate(tqdm(all_p)):
            img = imread(p, pilmode='L')
            name = path.split(p)[-1]
            mnt = np.array(mnt_reader(
                path.join(self.mnt_dir, f'{name[:-4]}.mnt')), dtype=np.float32)
            names.append(name)
            # e = imread(path.join(self.target_folder, name), pilmode='L')

            if self.transform:
                img = self.transform(img)
                # e = self.transform(e)

            imgs.append(img)
            #enhance.append(e)
            mnts.append(mnt)
            pathes.append(p)

        return imgs, mnts, pathes, names#, enhance

    def __getitem__(self, idx):

        return self.imgs[idx], self.names[idx], self.mnts[idx]



    def __len__(self):
        return len(self.imgs)

    @classmethod
    def NIST4_collate_fn(cls, batch):
        imgs, names, mnts = list(zip(*batch))

        stack = torch.stack if isinstance(imgs, torch.Tensor) else np.stack
        imgs = stack(imgs, 0)
        # ori = np.stack(ori, 0)
        # seg = np.stack(seg, 0)
        return dict(
            img = torch.from_numpy(imgs).float(),
            # ori = None,
            # seg = None,
            # pseg = None,
            # enh = None,
            path = names,
            mnt = mnts
        )



if __name__ == '__main__':

    from torch.utils.data import DataLoader
    # import cv2
    from matplotlib import pyplot as plt
    d = NIST4('/data/tangy/datasets/NIST4/general/L')
    l = DataLoader(d, 10, collate_fn=NIST4.NIST4_collate_fn)

    for i in l:

        ans = i
        print(ans['mnt'][0].shape, ans['img'].shape)

        break

    # for i in l:
    #     a = cv2.resize(np.squeeze(i[-1].numpy()), (800, 768), interpolation=cv2.INTER_NEAREST)
    #     print(a.shape)
    #     a = np.squeeze(i[0].numpy()) * a

    #     plt.imshow(a.astype(np.uint8))
    #     plt.savefig('1.png')
    #     break
