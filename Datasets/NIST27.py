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
import cv2

def T(img):
    return torch.from_numpy(np.expand_dims(img / 255., 0))

def imread(p, **args):
    return np.array(cv2.imread(p,flags=0), dtype=np.uint8)


class NIST27(Dataset):

    def __init__(self, base_path, transform=T, ori_dir='L_ori_manual', mnt_dir='L_M', target_folder='/Datasets/wzz/NISTSD27zx14/imgs/L', mode='a'):

        self.ori = {'G041L6U.bmp': '36.bmp', 'G071L2U.bmp': '63.bmp', 'G086L9U.bmp': '78.bmp', 'B121L7U.bmp': '107.bmp', 'U245L7U.bmp': '210.bmp', 'G068L6U.bmp': '60.bmp', 'B110L6U.bmp': '97.bmp', 'U256L7U.bmp': '220.bmp', 'G020L6U.bmp': '18.bmp', 'G077L1U.bmp': '69.bmp', 'G052L9U.bmp': '46.bmp', 'B190L8U.bmp': '163.bmp', 'B177L6U.bmp': '153.bmp', 'G089L0U.bmp': '81.bmp', 'G001L2U.bmp': '1.bmp', 'U214L5U.bmp': '183.bmp', 'G021L7U.bmp': '19.bmp', 'U272L9U.bmp': '234.bmp', 'G057L4U.bmp': '50.bmp', 'G046L1U.bmp': '41.bmp', 'U221L9U.bmp': '189.bmp', 'G060L1U.bmp': '53.bmp', 'U265L3U.bmp': '227.bmp', 'B109L4U.bmp': '96.bmp', 'G096L2U.bmp': '87.bmp', 'B197L8U.bmp': '170.bmp', 'U240L1U.bmp': '206.bmp', 'U213L5U.bmp': '182.bmp', 'U289L6U.bmp': '248.bmp', 'U237L1U.bmp': '204.bmp', 'U227L6U.bmp': '195.bmp', 'B193L2U.bmp': '166.bmp', 'G034L6U.bmp': '29.bmp', 'G053L6U.bmp': '47.bmp', 'U235L4U.bmp': '202.bmp', 'U225L6U.bmp': '193.bmp', 'G073L8U.bmp': '65.bmp', 'U251L1U.bmp': '215.bmp', 'B114L3U.bmp': '100.bmp', 'G062L9U.bmp': '55.bmp', 'U243L8U.bmp': '209.bmp', 'U252L6U.bmp': '216.bmp', 'B145L8U.bmp': '127.bmp', 'U280L5U.bmp': '239.bmp', 'U291L6U.bmp': '250.bmp', 'B154L6U.bmp': '135.bmp', 'U246L8U.bmp': '211.bmp', 'G017L9U.bmp': '15.bmp', 'G093L8U.bmp': '84.bmp', 'B141L4U.bmp': '123.bmp', 'B157L7U.bmp': '137.bmp', 'B105L2U.bmp': '92.bmp', 'B120L9U.bmp': '106.bmp', 'U261L6U.bmp': '225.bmp', 'G055L3U.bmp': '48.bmp', 'B163L7U.bmp': '140.bmp', 'G079L1U.bmp': '71.bmp', 'G038L6U.bmp': '33.bmp', 'U231L2U.bmp': '199.bmp', 'B173L9U.bmp': '150.bmp', 'B115L8U.bmp': '101.bmp', 'G072L1U.bmp': '64.bmp', 'B123L1U.bmp': '109.bmp', 'B140L8U.bmp': '122.bmp', 'B137L6U.bmp': '119.bmp', 'G019L1U.bmp': '17.bmp', 'G031L6U.bmp': '26.bmp', 'U233L6U.bmp': '201.bmp', 'B142L1U.bmp': '124.bmp', 'B144L3U.bmp': '126.bmp', 'B170L7U.bmp': '147.bmp', 'G030L1U.bmp': '25.bmp', 'G088L7U.bmp': '80.bmp', 'U285L4U.bmp': '244.bmp', 'B187L2U.bmp': '160.bmp', 'U298L2U.bmp': '256.bmp', 'B185L7U.bmp': '159.bmp', 'U223L6U.bmp': '191.bmp', 'B166L3U.bmp': '143.bmp', 'G004L8U.bmp': '4.bmp', 'U290L4U.bmp': '249.bmp', 'G065L6U.bmp': '58.bmp', 'U226L3U.bmp': '194.bmp', 'G009L8U.bmp': '9.bmp', 'U202L8U.bmp': '175.bmp', 'B135L8U.bmp': '117.bmp', 'B112L2U.bmp': '99.bmp', 'B161L1U.bmp': '139.bmp', 'U270L6U.bmp': '232.bmp', 'G007L1U.bmp': '7.bmp', 'U216L4U.bmp': '185.bmp', 'G081L1U.bmp': '73.bmp', 'G085L2U.bmp': '77.bmp', 'B155L6U.bmp': '136.bmp', 'U201L6U.bmp': '174.bmp', 'U204L2U.bmp': '176.bmp', 'U296L6U.bmp': '255.bmp', 'U253L9U.bmp': '217.bmp', 'B106L8U.bmp': '93.bmp', 'G075L3U.bmp': '67.bmp', 'G026L9U.bmp': '21.bmp', 'U220L5U.bmp': '188.bmp', 'U228L3U.bmp': '196.bmp', 'B139L2U.bmp': '121.bmp', 'U241L6U.bmp': '207.bmp', 'G033L7U.bmp': '28.bmp', 'U205L4U.bmp': '177.bmp', 'G028L1U.bmp': '23.bmp', 'U283L6U.bmp': '242.bmp', 'B200L6U.bmp': '173.bmp', 'U217L6U.bmp': '186.bmp', 'U206L3U.bmp': '178.bmp', 'B126L1U.bmp': '111.bmp', 'B199L1U.bmp': '172.bmp', 'U293L1U.bmp': '252.bmp', 'B183L1U.bmp': '158.bmp', 'U242L1U.bmp': '208.bmp', 'U271L6U.bmp': '233.bmp', 'B189L7U.bmp': '162.bmp', 'B104L8U.bmp': '91.bmp', 'G094L6U.bmp': '85.bmp', 'G069L7U.bmp': '61.bmp', 'U215L8U.bmp': '184.bmp', 'G078L3U.bmp': '70.bmp', 'B152L8U.bmp': '133.bmp', 'U209L7U.bmp': '179.bmp', 'U263L1U.bmp': '226.bmp', 'G049L3U.bmp': '43.bmp', 'U258L1U.bmp': '222.bmp',
                    'G002L3U.bmp': '2.bmp', 'U284L2U.bmp': '243.bmp', 'G023L6U.bmp': '20.bmp', 'G090L5U.bmp': '82.bmp', 'U300L2U.bmp': '258.bmp', 'B180L4U.bmp': '155.bmp', 'G070L2U.bmp': '62.bmp', 'G016L8U.bmp': '14.bmp', 'U254L3U.bmp': '218.bmp', 'G043L8U.bmp': '38.bmp', 'G084L6U.bmp': '76.bmp', 'B111L7U.bmp': '98.bmp', 'G032L6U.bmp': '27.bmp', 'G076L3U.bmp': '68.bmp', 'U276L6U.bmp': '237.bmp', 'G027L6U.bmp': '22.bmp', 'B147L7U.bmp': '129.bmp', 'U259L7U.bmp': '223.bmp', 'B179L2U.bmp': '154.bmp', 'G087L8U.bmp': '79.bmp', 'G047L4U.bmp': '42.bmp', 'U299L8U.bmp': '257.bmp', 'G051L6U.bmp': '45.bmp', 'G003L8U.bmp': '3.bmp', 'B130L1U.bmp': '114.bmp', 'B194L8U.bmp': '167.bmp', 'B175L6U.bmp': '151.bmp', 'U255L1U.bmp': '219.bmp', 'B192L9U.bmp': '165.bmp', 'B158L8U.bmp': '138.bmp', 'U212L2U.bmp': '181.bmp', 'U249L3U.bmp': '213.bmp', 'U267L3U.bmp': '229.bmp', 'U274L9U.bmp': '236.bmp', 'B172L1U.bmp': '149.bmp', 'B169L2U.bmp': '146.bmp', 'B107L9U.bmp': '94.bmp', 'B129L7U.bmp': '113.bmp', 'G063L2U.bmp': '56.bmp', 'U222L3U.bmp': '190.bmp', 'U294L6U.bmp': '253.bmp', 'U229L2U.bmp': '197.bmp', 'U266L3U.bmp': '228.bmp', 'B196L7U.bmp': '169.bmp', 'G040L6U.bmp': '35.bmp', 'G064L1U.bmp': '57.bmp', 'B117L0U.bmp': '103.bmp', 'G013L4U.bmp': '12.bmp', 'G074L3U.bmp': '66.bmp', 'U287L3U.bmp': '246.bmp', 'B146L6U.bmp': '128.bmp', 'B153L1U.bmp': '134.bmp', 'U247L1U.bmp': '212.bmp', 'B167L2U.bmp': '144.bmp', 'G095L6U.bmp': '86.bmp', 'G059L1U.bmp': '52.bmp', 'G092L3U.bmp': '83.bmp', 'U236L2U.bmp': '203.bmp', 'G018L6U.bmp': '16.bmp', 'B138L3U.bmp': '120.bmp', 'B134L9U.bmp': '116.bmp', 'B150L3U.bmp': '131.bmp', 'B188L7U.bmp': '161.bmp', 'B102L0U.bmp': '90.bmp', 'B136L8U.bmp': '118.bmp', 'G035L6U.bmp': '30.bmp', 'B118L8U.bmp': '104.bmp', 'U269L3U.bmp': '231.bmp', 'B168L1U.bmp': '145.bmp', 'U282L2U.bmp': '241.bmp', 'B132L7U.bmp': '115.bmp', 'U292L1U.bmp': '251.bmp', 'B164L3U.bmp': '141.bmp', 'U211L8U.bmp': '180.bmp', 'G012L8U.bmp': '11.bmp', 'B122L4U.bmp': '108.bmp', 'G099L9U.bmp': '88.bmp', 'U277L9U.bmp': '238.bmp', 'B151L9U.bmp': '132.bmp', 'U286L6U.bmp': '245.bmp', 'G014L7U.bmp': '13.bmp', 'B119L0U.bmp': '105.bmp', 'G058L4U.bmp': '51.bmp', 'G082L6U.bmp': '74.bmp', 'G061L4U.bmp': '54.bmp', 'G039L4U.bmp': '34.bmp', 'B191L3U.bmp': '164.bmp', 'U268L4U.bmp': '230.bmp', 'B108L6U.bmp': '95.bmp', 'B181L8U.bmp': '156.bmp', 'U218L1U.bmp': '187.bmp', 'B148L6U.bmp': '130.bmp', 'U224L6U.bmp': '192.bmp', 'B124L5U.bmp': '110.bmp', 'U232L4U.bmp': '200.bmp', 'B127L6U.bmp': '112.bmp', 'G045L9U.bmp': '40.bmp', 'U257L3U.bmp': '221.bmp', 'G083L8U.bmp': '75.bmp', 'U250L1U.bmp': '214.bmp', 'G005L8U.bmp': '5.bmp', 'G029L9U.bmp': '24.bmp', 'B165L7U.bmp': '142.bmp', 'B176L7U.bmp': '152.bmp', 'G011L7U.bmp': '10.bmp', 'G042L2U.bmp': '37.bmp', 'B143L6U.bmp': '125.bmp', 'G006L6U.bmp': '6.bmp', 'U230L6U.bmp': '198.bmp', 'B171L7U.bmp': '148.bmp', 'U273L6U.bmp': '235.bmp', 'B195L3U.bmp': '168.bmp', 'U295L1U.bmp': '254.bmp', 'B116L6U.bmp': '102.bmp', 'G080L8U.bmp': '72.bmp', 'B101L9U.bmp': '89.bmp', 'G036L6U.bmp': '31.bmp', 'G056L9U.bmp': '49.bmp', 'U281L7U.bmp': '240.bmp', 'B198L6U.bmp': '171.bmp', 'G008L6U.bmp': '8.bmp', 'U238L1U.bmp': '205.bmp', 'U260L8U.bmp': '224.bmp', 'G066L4U.bmp': '59.bmp', 'B182L7U.bmp': '157.bmp', 'U288L6U.bmp': '247.bmp', 'G050L7U.bmp': '44.bmp', 'G037L6U.bmp': '32.bmp', 'G044L7U.bmp': '39.bmp'}
        self.ori_dir = path.join(path.split(base_path)[0], ori_dir)
        self.mnt_dir = path.join(path.split(base_path)[0], mnt_dir)
        # print(self.ori_dir)
        self.base_path = base_path
        self.transform = transform
        self.target_folder = target_folder
        assert mode in ['a', 's', 'm'] # a-all s-single m-muiliti
        self.mode = mode
        self.name_f = []
        if mode != 'a':
            with open(path.join('/data3/albert/NISTSD27zx14', 'single_name_list.json'), 'r') as f:
                self.name_f = json.load(f)
        self.imgs, self.ori_labels, self.seg_labels, self.mnts, self.path, self.names, self.enhance, self.pred_seg = self.get_all_images()

    def get_all_images(self):

        def transTxt2array(txt_path):

            cont = open(txt_path).readlines()[1:]
            cont = [elem.split(' ')[:-1] for elem in cont]
            cont = np.array(cont).astype('float')

            seg = np.ones_like(cont)
            seg[cont > 90] = 0
            return cont.astype('float'), seg

        all_p = glob(f'{self.base_path}/*')
        if self.mode == 's': all_p = list(filter(lambda x : x.split('/')[-1] in self.name_f, all_p))
        if self.mode == 'm': all_p = list(filter(lambda x : x.split('/')[-1] not in self.name_f, all_p))
        imgs = []
        ori_labels = []
        seg_labels = []
        pathes = []
        mnts = []
        names = []
        enhance = []
        pre_seg = []


        for i, p in enumerate(tqdm(all_p)):
            img = imread(p, pilmode='L')

            name = path.split(p)[-1]
            label = path.join(self.ori_dir, f'bd{self.ori[name][:-4]}.txt')
            mnt = np.array(mnt_reader(
                path.join(self.mnt_dir, f'{name[:-4]}.mnt')), dtype=np.float32)
            olabel, slabel = transTxt2array(label)

            
            ori_labels.append(olabel)
            seg_labels.append(slabel)
            names.append(name)
            e = imread(path.join(self.target_folder, name), pilmode='L')
            pseg = imread(path.join(self.target_folder.replace('imgs', 'seg'), name), pilmode='L')

            if self.transform:
                img = self.transform(img)
                e = self.transform(e)
                pseg = self.transform(pseg)

            imgs.append(img)
            enhance.append(e)
            pre_seg.append(pseg)
            # w, h = img.shape
            # row, col = (mnt[:, 1] // 8).astype(np.int32), (mnt[:, 0] // 8).astype(np.int32)
            # mnt_mat = np.zeros((3, w // 8, h // 8))
            # mnt_mat[0, row, col] = mnt[:, 1]
            # mnt_mat[1, row, col] = mnt[:, 0]
            # mnt_mat[2, row, col] = mnt[:, 2]
            mnts.append(mnt)

            pathes.append(p)

        return imgs, ori_labels, seg_labels, mnts, pathes, names, enhance, pre_seg

    def __getitem__(self, idx):

        return self.imgs[idx], self.ori_labels[idx], self.seg_labels[idx],  self.names[idx], self.enhance[idx], self.pred_seg[idx], self.mnts[idx]



    def __len__(self):
        return len(self.imgs)

    @classmethod
    def NIST27_collate_fn(cls, batch):
        imgs, ori, seg, name, enhance, pseg, mnts = list(zip(*batch))

        stack = torch.stack if isinstance(imgs, torch.Tensor) else np.stack
        imgs = stack(imgs, 0)
        ori = np.stack(ori, 0)
        seg = np.stack(seg, 0)
        pseg = np.stack(pseg, 0)
        enhance = np.stack(enhance, 0)
        return dict(
            img = torch.from_numpy(imgs).float(),
            ori = ori,
            seg = seg,
            pseg = pseg,
            enh = enhance,
            path = name,
            mnt = mnts
        )



if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import cv2
    from matplotlib import pyplot as plt
    d = NIST27('/Datasets/NISTSD27/matched/L')
    l = DataLoader(d, 10)

    for i in l:

        imgs, _, _, mnt = i
        print(mnt.shape)

        break

    # for i in l:
    #     a = cv2.resize(np.squeeze(i[-1].numpy()), (800, 768), interpolation=cv2.INTER_NEAREST)
    #     print(a.shape)
    #     a = np.squeeze(i[0].numpy()) * a

    #     plt.imshow(a.astype(np.uint8))
    #     plt.savefig('1.png')
    #     break
