import torch
from PostProcessing.MinutiaeTools import MinutiaeTools
from PostProcessing.Morph import Erode, Dilate
from PostProcessing.EnhanceImageProducer import EnhanceImageProducer
from PostProcessing.SegmentLabelTools import SegmentLabelTools

from FingerNet.Minutiae import Minutiae
from FingerNet.Orientation import Orientation
from utils import load_checkpoint



class FingerNet(torch.nn.Module):

    def __init__(self, dev=0):
        super(FingerNet, self).__init__()
        self.dev = torch.device(dev)
        self.threhold = 0.45

        self.Orientation = Orientation()
        self.phaModel = Minutiae()
        load_checkpoint('fingero.pth', self.Orientation)
        load_checkpoint('fingerm.pth', self.phaModel)

        self.mt = MinutiaeTools()
        self.label = SegmentLabelTools()
        self.producer = EnhanceImageProducer()
        self.dilate = Dilate(5)


    def forward(self, imgs, finger_type='L', path=None):
        ori, seg, TF, OF = self.Orientation(imgs)
        enhanceImageP = self.producer(imgs.float(), ori.float())

        segment = torch.round(seg)
        segment = self.dilate(segment).float()

        segment_big = torch.nn.functional.interpolate(segment.float(), scale_factor=8, recompute_scale_factor=True)
        enhanceImage = enhanceImageP# * segment_big

        mi = torch.min(enhanceImage.reshape(enhanceImage.size(0), -1), 1).values.reshape(enhanceImage.size(0), 1, 1, 1)
        mx = torch.max(enhanceImage.reshape(enhanceImage.size(0), -1), 1).values.reshape(enhanceImage.size(0), 1, 1, 1)
        enhanceImage_normlized = ((enhanceImage - mi) / (mx - mi + 1e-6))

        confidencePHA, w, h, o = self.phaModel(enhanceImageP, TF, OF, ori, seg)

        c = confidencePHA
        confidenceO, confidenceC = self.mt.NMS_adaptive(c, segment, w, h, o, self.threhold, 15, 0.15)
        mnt_numbers = torch.count_nonzero(confidenceO.reshape(imgs.shape[0], -1), -1)  # 用来区分每一个图像的细节点
        mnt_spareO = confidenceO.to_sparse().coalesce()
        mnt_spareC = confidenceC.to_sparse().coalesce()
        mnt = torch.stack((mnt_spareC.indices()[3], mnt_spareC.indices()[2], mnt_spareO.values(), mnt_spareC.values()), dim=0).T
    

        out = dict(
            batch=imgs.size(0), 
            file_path=path, 
            size=(imgs.size(2), imgs.size(3)),
            mnt=mnt, 
            enhanceImage = enhanceImage_normlized,
            mnt_numbers=mnt_numbers, 
            segment=segment,
            seg=seg,
            ori = ori,
            confidence = torch.cat((c, w, h, o), 1)
        )
        return out

