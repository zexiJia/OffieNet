import torch
from utils import PRC_new
import numpy as np

pi = 3.141592653589793

class MinutiaeTools:

    def __init__(self, fp_16=False, r=8):

        if fp_16:
            self.NMS= self.NMS_fp16
        else:
            self.NMS = self.NMS_exp
        
        self.r = r
    

    @torch.no_grad()
    def NMS_exp(self, confidence, segment, wp, hp, ori, threhold):
        
        if segment.ndim == 3: torch.unsqueeze(segment, dim=1)
        segment = torch.where(segment < 0.5, 0, 1)

        batch, channel, w, h = confidence.shape
        confidence_big = torch.zeros((batch, channel, w*8, h*8), device=confidence.device)
        confidence = confidence * segment
        position = torch.nonzero(confidence> threhold, as_tuple=True)

        wp = torch.floor(wp * 7).long()
        hp = torch.floor(hp * 7).long()
        ori = torch.atan2(ori[:, 0:1, ...], ori[:, 1:, ...])
        wp = wp[position[0], position[1], position[2], position[3]]
        hp = hp[position[0], position[1], position[2], position[3]]
        ori = ori[position[0], position[1], position[2], position[3]]
        ori[ori < 0 ] += 2 * pi
        ori[ori == 0] += 1e-6

        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = confidence[[position[0], position[1], position[2], position[3]]]
        confidence_candicate = torch.nn.functional.max_pool2d(confidence_big, kernel_size=(self.r * 2 + 1, self.r * 2 + 1), stride=(1, 1), padding=self.r)
        confidence_candicate = confidence_big - confidence_candicate
        confidence_candicate = torch.where(confidence_candicate < 0, 0, 1)
        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = ori
        confidence_ori = confidence_candicate * confidence_big # 有数的地方有方向

        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = confidence[[position[0], position[1], position[2], position[3]]]
        confidence = confidence_candicate * confidence_big # 有数的地方有置信度
        return confidence_ori, confidence


    @torch.no_grad()
    def NMS_fp16(self, confidence, segment, wp, hp, ori, threhold):
        if segment.ndim == 3: torch.unsqueeze(segment, dim=1)
        segment = torch.round(segment)# < threhold, 0, 1)

        batch, channel, w, h = confidence.shape
        confidence_big = torch.zeros((batch, channel, w*8, h*8), device=confidence.device).half()
        confidence = confidence * segment
        position = torch.nonzero(confidence> threhold, as_tuple=True)

        wp = torch.floor(wp * 7).long()
        hp = torch.floor(hp * 7).long()
        ori = torch.atan2(ori[:, 0:1, ...], ori[:, 1:, ...])
        wp = wp[position[0], position[1], position[2], position[3]]
        hp = hp[position[0], position[1], position[2], position[3]]
        ori = ori[position[0], position[1], position[2], position[3]]
        ori[ori < 0 ] += 2 * pi
        ori[ori == 0] += 1e-6

        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = confidence[[position[0], position[1], position[2], position[3]]]
        confidence_candicate = torch.nn.functional.max_pool2d(confidence_big, kernel_size=(self.r * 2 + 1, self.r * 2 + 1), stride=(1, 1), padding=self.r)
        confidence_candicate = confidence_big - confidence_candicate
        confidence_candicate = torch.where(confidence_candicate < 0, 0, 1)
        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = ori.half()
        confidence_ori = confidence_candicate * confidence_big # 有数的地方有方向

        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = confidence[[position[0], position[1], position[2], position[3]]]
        confidence = confidence_candicate * confidence_big # 有数的地方有置信度
        return confidence_ori, confidence


    def extract_mnt_four_tuple(self, confidence_candicate, confidence):

        indexo = confidence_candicate.to_sparse().coalesce()
        indexc = confidence.to_sparse().coalesce()
        return torch.stack((indexo.indices()[3], indexo.indices()[2], indexo.values(), indexc.values()), dim=0).T

    def extract_mnt(self, confidence_candicate):

        indexo = confidence_candicate.to_sparse().coalesce()
        return torch.stack((indexo.indices()[3], indexo.indices()[2], indexo.values()), dim=0).T


    def NMS_adaptive(self, confidence, segment, wp, hp, ori, threhold, at_least=15, low=0.2):
        batch = confidence.size(0)
        threhold = torch.tensor([threhold]).repeat(batch, 1, 1, 1).cuda().half()
        confidenceO, confidenceC = self.NMS(confidence, segment, wp, hp, ori, threhold)

        # 保证至少提出10个细节点
        res = torch.count_nonzero(( confidenceO).reshape(batch, -1), dim=-1)
        is_q = res < at_least
        while torch.count_nonzero(is_q).item() > 0:# and threhold.min().item() > low:
            threhold[is_q] -= 0.01
            confidenceO, confidenceC =  self.NMS(confidence, segment, wp, hp, ori, threhold)
            res = torch.count_nonzero(( confidenceO).reshape(batch, -1), dim=-1)
            is_q = torch.logical_and((res < at_least).reshape(batch, 1, 1, 1), threhold > low)
        
        # print(threhold.reshape(batch))
        return confidenceO, confidenceC


    @torch.no_grad()
    def PRC_metirc(self, mnt_truth, mnt_pred, name=None):
        
        p = []
        r = []
        for i in range(len(mnt_truth)):

            rp = PRC_new(mnt_truth[i], mnt_pred[i])
            p.append(rp[0])
            r.append(rp[1])
            if p[-1] < 0.5 or r[-1] < 0.5:
                print(f'{p[-1]}, {r[-1]}, \033[0;31m {name[i][0]} \033[0m')
                continue
            print(f'{p[-1]}, {r[-1]}, \033[0m {name[i][0]} \033[0m')
        
        p = np.array(p).mean()
        r = np.array(r).mean()

        return p, r, p+r

    
