from PostProcessing.MinutiaeTools import MinutiaeTools

from FingerNet.Minutiae import Minutiae
from FingerNet.Orientation import Orientation
from utils import load_checkpoint

import torch
import numpy as np


class PalmNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.Orientation = Orientation()
        self.phaModel = Minutiae()
        self.ampModel = Minutiae()

        load_checkpoint('/home/albert/PalmNetExp/PalmX/pytorch_weights/palmo.pth', self.Orientation)
        load_checkpoint('/home/albert/PalmNetExp/PalmX/pytorch_weights/amp.pth', self.ampModel)
        load_checkpoint('/home/albert/PalmNetExp/PalmX/pytorch_weights/pha.pth', self.phaModel)


        self.Orientation = self.Orientation.to(self.dev)
        self.phaModel = self.phaModel.to(self.dev)
        self.ampModel = self.ampModel.to(self.dev)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.mt = MinutiaeTools()


    def forward(self, imgs, path=None):

        if isinstance(imgs, np.ndarray):
            assert imgs.ndim != 3, "如果输入是单个图像(imgs), 请保持图像符合(h, w), 而不是(c, h, w) 或 (h, w, c)"
            imgs = torch.tensor(imgs)
        if isinstance(imgs, list):
            assert imgs[0].ndim != 3, "如果输入是一组图像(imgs), 请保持每个图像符合(h, w), 而不是(c, h, w) 或 (h, w, c)"
            imgs = np.stack(imgs, axis=0)
            imgs = torch.tensor(imgs)
        
        if imgs.ndim == 2:
            imgs = imgs.reshape(1, 1, imgs.size(0), imgs.size(1))
        if imgs.ndim == 3:
            imgs = imgs.reshape(imgs.size(0), 1, imgs.size(1), imgs.size(2))

        imgs = imgs.to(self.dev)
        imgs = imgs / 255.
        imgs = (imgs - 0.5) / 0.5
        paddingh, paddingw = (32 - (imgs.size(2)) % 32) % 32, (32 - (imgs.size(3)) % 32) % 32
        imgs = torch.nn.functional.pad(imgs, (0, paddingw, 0, paddingh))
        ori, seg, TF, OF, QF = self.Orientation(imgs)
        pha, amp, enhanceImage = self.Gaborx.GaborX(imgs.float(), ori.float(), 8)

        segment = torch.round(seg)
        segment = self.MorphOp.cuDilate(segment.bool(), 5)
        segment = self.MorphOp.cuErode(segment.bool(), 5).to(seg.dtype)
 
        mask = self.segmentpostMask(segment.bool()).to(seg.dtype)

        segment_big = torch.nn.functional.interpolate(segment.float(), scale_factor=8, recompute_scale_factor=True)
        enhanceImage = enhanceImage * segment_big
        phaI = pha * segment_big
        ampI = amp * segment_big

        confidencePHA, w, h, o = self.phaModel(pha, TF, OF, ori, segment)
        confidenceAMP, _, _, _ = self.ampModel(amp, TF, OF, ori, segment)
        
        c = 0.5 * (confidencePHA + confidenceAMP)
        confidenceO, confidenceC = self.mt.NMS_adaptive(c, segment, w, h, o, 0.45, 15, 0.15)
        mnt_numbers = torch.count_nonzero(confidenceO.reshape(imgs.shape[0], -1), -1)  # 用来区分每一个图像的细节点
        mnt_spareO = confidenceO.to_sparse().coalesce()
        mnt_spareC = confidenceC.to_sparse().coalesce()
        mnt = torch.stack((mnt_spareC.indices()[3], mnt_spareC.indices()[2], mnt_spareO.values(), mnt_spareC.values()), dim=0).T


        out = dict(
            batch = imgs.size(0), 
            file_path = path, 
            size = (imgs.size(2), imgs.size(3)),
        
            enhance = enhanceImage.cpu().numpy(), 
            mask = mask.cpu().numpy(),

            mnt = mnt.cpu().numpy(), 
            mnt_numbers = mnt_numbers.cpu().numpy(), 
            segment = segment.cpu().numpy(),
            seg = seg.cpu().numpy(),
  
            ori = ori.cpu().numpy(),
            confidence = torch.cat((c, w, h, o), 1).cpu().numpy(),
            amp = ampI.cpu().numpy(),
            pha = phaI.cpu().numpy()
        )
        
        return out

    def segmentpostMask(self, segment):
        mask = torch.zeros_like(segment, dtype=torch.float16)
        segment1 = self.MorphOp.cuDilate(segment.bool(), 1)
        segment2 = self.MorphOp.cuDilate(segment.bool(), 3)
        segment3 = self.MorphOp.cuDilate(segment.bool(), 5)
        segment4 = self.MorphOp.cuDilate(segment.bool(), 7)
        segment5 = self.MorphOp.cuDilate(segment.bool(), 9)
        mask[segment1] = 1.
        mask[torch.logical_xor(segment1, segment2)] = 0.9
        mask[torch.logical_xor(segment2, segment3)] = 0.8
        mask[torch.logical_xor(segment3, segment4)] = 0.7
        mask[torch.logical_xor(segment4, segment5)] = 0.5
        return mask
