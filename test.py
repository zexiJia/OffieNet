import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

from pathlib import Path

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from datasets import DataAugmentationForMAE
from Datasets.ImageFolder import Images
from FingerNet.FingerNet import FingerNet
from Models.basenet import BaseNet
from Models.offienet import OffieNet
from ckputils import checkpoint
import torch.nn.functional as F  
import cv2
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(
        "OffieNet visualization reconstruction script", add_help=False
    )

    parser.add_argument(
        "--input_size", default=512, type=int, help="images input size for backbone"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="device to use for training / testing"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="OffieNet",
        type=str,
        metavar="MODEL",
        help="Name of model to vis",
    )

    return parser.parse_args()


def zero_random_rect(images, mask=0, min_size=50, max_size=150):  
    """  
    Sets the RGB values within a random rectangular region to 0 in the given batch of images.  
    """  
    batch_size, channels, height, width = images.size()  
      
    for i in range(batch_size):  
        # Randomly select the top-left corner of the rectangle  
        top_left_x = random.randint(width//2 - min_size//2 , width//2 + min_size//2)  
        top_left_y = random.randint(height//2 - min_size//2, height//2 + min_size//2)  
          
        # Randomly select the width and height of the rectangle  
        rect_width = random.randint(min_size, min(max_size, width - top_left_x))  
        rect_height = random.randint(min_size, min(max_size, height - top_left_y))  
          
        # Select a rectangular region in the image and set its RGB values to 0
        if mask == 0:
            images[i, :, top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width] = 0
            return images
        elif mask == 1:
            mask_images = torch.zeros_like(images) 
            mask_images[i, :, top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width] = images[i, :, top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width]
            return mask_images
    


def main(args):

    device = torch.device(args.device)
    cudnn.benchmark = True

    if args.model == "BaseNet":
        autoencoder = BaseNet()
        checkpoint_path = "basenet.pth"
        checkpoints = torch.load(checkpoint_path)
        autoencoder.load_state_dict(checkpoints["net_weights"])
        autoencoder.eval()
        autoencoder.to(device)
    
    elif args.model == "OffieNet":
        autoencoder = OffieNet()
        checkpoint_path = "offienet.pth"
        checkpoints = torch.load(checkpoint_path)
        autoencoder.load_state_dict(checkpoints["net_weights"])
        autoencoder.eval()
        autoencoder.to(device)



    test_loader = torch.utils.data.DataLoader(
        Images(["/Datasets/FVCDataset/FVC2004/DB1_a"], None, None),
        1,
        True,
        num_workers=1,
        pin_memory=True,
    )


    seg_dataset = Images(["/Datasets/NISTSD27/matched/seg"], None, None)

    fnet = FingerNet().cuda()
    fnet.eval()
    checkpoint("fingernet.pth", "net_weights").keep(
        "net", True
    ).load_ckp(fnet, strict=False)

    mpsnr = 0

    if args.model == "BaseNet":

        for i, data in enumerate(test_loader):
            imgs, oid = data
            imgs = imgs.float().cuda()


            with torch.no_grad():
                ans = fnet((imgs - 0.5) / 0.5)
                ori = ans["ori"]
                enh = ans["enhanceImage"]

                imgs = zero_random_rect(imgs,0,50,100)
                outputs = autoencoder(imgs)

                outputs = outputs.squeeze().detach().cpu().numpy()
                imgs = imgs.squeeze().detach().cpu().numpy()
                enh = enh.squeeze().detach().cpu().numpy()

                outputs = outputs * 255
                outputs = outputs.astype(np.int16)

                enh = enh * 255
                enh = enh.astype(np.int16)
                imgs = imgs * 255
                imgs = imgs.astype(np.int16)

                ids = oid[0].split("/")[-1]

                vis = np.concatenate((imgs, enh, outputs), 1)

                cv2.imwrite('vis_base/'+ids, vis)
    
    
    elif args.model == "OffieNet":
        for i, data in enumerate(test_loader):
            imgs, oid = data
            imgs = imgs.float().cuda()


            with torch.no_grad():
                ans = fnet((imgs - 0.5) / 0.5)
                ori = ans["ori"]
                enh = ans["enhanceImage"]
                ori = F.interpolate(ori, scale_factor=8, mode='bilinear', align_corners=False) 

                imgs = zero_random_rect(imgs,0,50,100)
                
                outputs,outputs_ori = autoencoder(imgs,ori)

                outputs = outputs.squeeze().detach().cpu().numpy()
                imgs = imgs.squeeze().detach().cpu().numpy()
                enh = enh.squeeze().detach().cpu().numpy()

                outputs = outputs * 255
                outputs = outputs.astype(np.int16)

                enh = enh * 255
                enh = enh.astype(np.int16)
                imgs = imgs * 255
                imgs = imgs.astype(np.int16)

                ids = oid[0].split("/")[-1]

                vis = np.concatenate((imgs, enh, outputs), 1)

                cv2.imwrite('vis_offie/'+ids, vis)






if __name__ == "__main__":
    opts = get_args()
    main(opts)
