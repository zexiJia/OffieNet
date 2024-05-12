import imp
import torch
from utils import PRC_new
import numpy as np

pi = 3.141592653589793

class MinutiaeTools:

    def __init__(self):
        self.NMS = self.NMS_exp
    

    @torch.no_grad()
    def NMS_exp(self, confidence, segment, wp, hp, ori, threhold):
        # print(wp)
        segment[segment < threhold] = 0
        segment[segment >= threhold] = 1

        batch, channel, w, h = confidence.shape
        confidence_big = torch.zeros((batch, channel, w*8, h*8), device=confidence.device)
        position = torch.nonzero((confidence * segment)> threhold, as_tuple=True)

        wp = torch.floor(wp * 7).long()
        hp = torch.floor(hp * 7).long()
        ori = torch.atan2(ori[:, 0:1, ...], ori[:, 1:, ...])
        wp = wp[position[0], position[1], position[2], position[3]]
        hp = hp[position[0], position[1], position[2], position[3]]
        ori = ori[position[0], position[1], position[2], position[3]]
        ori[ori < 0 ] += 2 * pi
        ori[ori == 0] += 1e-5

        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = confidence[[position[0], position[1], position[2], position[3]]]
        confidence_candicate = torch.nn.functional.max_pool2d(confidence_big, kernel_size=(17, 17), stride=(1, 1), padding=8)
        confidence_candicate = confidence_big - confidence_candicate
        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = ori.float()

        confidence_candicate = torch.where(confidence_candicate < 0, 0, 1) * confidence_big
        return confidence_candicate

    @torch.no_grad()
    def NMS_adaptive(self, confidence, segment, wp, hp, ori, threhold, at_least=15, low=0.2):
        batch = confidence.size(0)
        threhold = torch.tensor([threhold]).repeat(batch, 1, 1, 1).cuda()
        confidenceO = self.NMS_exp(confidence, segment, wp, hp, ori, threhold)

        # 保证至少提出10个细节点
        res = torch.count_nonzero(( confidenceO).reshape(batch, -1), dim=-1)
        is_q = res < at_least
        while torch.count_nonzero(is_q).item() > 0:# and threhold.min().item() > low:
            threhold[is_q] -= 0.01
            confidenceO =  self.NMS_exp(confidence, segment, wp, hp, ori, threhold)
            res = torch.count_nonzero(( confidenceO).reshape(batch, -1), dim=-1)
            is_q = torch.logical_and((res < at_least).reshape(batch, 1, 1, 1), threhold > low)
        
        # print(threhold.reshape(batch))
        return confidenceO


    def extract_mnt(self, confidence_candicate):

        confidence_candicate = torch.squeeze(confidence_candicate)
        index = confidence_candicate.to_sparse().coalesce()
        return torch.stack((index.indices()[1], index.indices()[0], index.values()), dim=0).T


    @torch.no_grad()
    def PRC_metirc(self, mnt_truth, mnt_pred):
        p = []
        r = []
        for i in range(len(mnt_truth)):

           rp = PRC_new(mnt_truth[i], mnt_pred[i])
           p.append(rp[0])
           r.append(rp[1])
           
        
        p = np.array(p).mean()
        r = np.array(r).mean()

        return p, r, p+r

    
