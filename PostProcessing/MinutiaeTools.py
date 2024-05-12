import torch
from utils import PRC_new
import numpy as np

pi = 3.141592653589793

class MinutiaeTools:

    def __init__(self, r=8):
        self.r = r


    @torch.no_grad()
    def NMS(self, confidence, segment, wp, hp, ori, threhold):
        assert confidence.dtype == segment.dtype and segment.dtype == wp.dtype and wp.dtype == hp.dtype and hp.dtype == ori.dtype, '输入的所有特征类型 must be equal'
        dtype = confidence.dtype

        if segment.ndim == 3: torch.unsqueeze(segment, dim=1)
        segment = torch.round(segment)# < threhold, 0, 1)

        batch, channel, w, h = confidence.shape
        confidence_big = torch.zeros((batch, channel, w*8, h*8), device=confidence.device, dtype=dtype)
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
        confidence_big[position[0], position[1], position[2]*8+hp, position[3]*8+wp] = ori.to(dtype)
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
        assert confidence.dtype == segment.dtype and segment.dtype == wp.dtype and wp.dtype == hp.dtype and hp.dtype == ori.dtype, f'输入的所有特征类型 must be equal {confidence.dtype} {segment.dtype}'
        dtype = confidence.dtype

        batch = confidence.size(0)
        threhold = torch.tensor([threhold]).repeat(batch, 1, 1, 1).to(confidence.device).to(dtype)
        confidenceO, confidenceC = self.NMS(confidence, segment, wp, hp, ori, threhold)

        # 保证至少提出10个细节点
        res = torch.count_nonzero(( confidenceO).reshape(batch, -1), dim=-1)
        is_q = res < at_least
        while torch.count_nonzero(is_q).item() > 0:# and threhold.min().item() > low:
            threhold[is_q] -= 0.01
            confidenceO, confidenceC =  self.NMS(confidence, segment, wp, hp, ori, threhold)
            res = torch.count_nonzero(( confidenceO).reshape(batch, -1), dim=-1)
            is_q = torch.logical_and((res < at_least).reshape(batch, 1, 1, 1), threhold > low)

        return confidenceO, confidenceC


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
