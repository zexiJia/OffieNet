import numpy as np
import torch
import torch.nn.functional as F
from PostProcessing.SegmentLabelTools import SegmentLabelTools
pi = np.pi


class OrientationTools:

    def __init__(self, dev=None):

        self.Guassian = torch.nn.Conv2d(2, 2, (5, 5), groups=2, padding=2, bias=False)
        sigma = 0.3*((5-1)*0.5 - 1) + 0.8 #opencv 自动计算公式
        self.Guassian.weight = torch.nn.Parameter(self.gaussian_kernel(sigma = sigma).repeat((2, 1,1,1)))

        self.angle_kernel = torch.arange(1, 180, 2, dtype=torch.float32).reshape((1, 90, 1, 1)) / 90. * pi

        if dev:
            self.Guassian.to(dev)
            self.angle_kernel = self.angle_kernel.to(dev)
        self.dev = dev
        self.label = SegmentLabelTools(0.5)



    def downsample(self, ori, seg, thre=0.5):

        if isinstance(ori, np.ndarray):
            ori = torch.tensor(ori)

        if isinstance(seg, np.ndarray):
            seg = torch.tensor(seg)
        
        batch = ori.shape[0]
        rad = self.get_rad(ori)

        sin_2angle = torch.sin(2 * rad)
        cos_2angle = torch.cos(2 * rad)

        sin_2angle = F.avg_pool2d(sin_2angle, kernel_size=2, stride=2)
        cos_2angle = F.avg_pool2d(cos_2angle, kernel_size=2, stride=2)
        angle = torch.atan2(sin_2angle, cos_2angle)

        if seg is None:
            return angle / 2 , None
        seg[seg < thre] = 0
        # seg = F.max_pool2d(seg, kernel_size=2, stride=2)
        seg = F.interpolate(seg, scale_factor= 0.5, recompute_scale_factor=True)

        return angle / 2 , torch.unsqueeze(seg, dim=1)


    def get_rad(self, ori):

        if ori.dim() == 3 or ori.shape[1] == 1:
            return ori

        sin2theta = ori[:, 0, ...]
        cos2theta = ori[:, 1, ...]

        return torch.atan2(sin2theta, cos2theta) / 2


    def test_ori_on_an_image(self, y_true, y_pred, seg, seg_prd=None):

        seg = seg if seg_prd is None else seg_prd

        #* 计算误差的时 180° 为一个周期, 例如-90°-90°=180°, 但其实为一个角度， 又或者 -90° 与 89° 直接相减为179°，其实相差 1° (1.7 support 👇)
        loss = torch.minimum(torch.abs(y_true[seg > 0] - y_pred[seg > 0]), 180 - torch.abs((y_true[seg > 0] - y_pred[seg > 0])))
        
        # # 1.6 support
        # sub, sub_round = (y_true[seg > 0] - y_pred[seg_prd > 0]), (180 - (y_true[seg > 0] - y_pred[seg_prd > 0]))
        # loss = torch.where(torch.abs(sub) >= torch.abs(sub_round), sub_round, sub)

        return torch.sqrt((loss ** 2).mean())



    def test_ori_on_a_batch(self, y_true, y_pred, seg, seg_prd=None):
      
        seg = seg if seg_prd is None else seg_prd
        batch_size = seg.shape[0]

        # print(loss.shape)
        # print(y_pred.shape, seg.shape)
        y_true = y_true * seg
        y_pred = y_pred * seg

        seg_num = torch.count_nonzero(seg.reshape(batch_size, -1), dim=-1)
        sub = y_true - y_pred
        loss = torch.minimum(torch.abs(sub), 180 - torch.abs(sub))
        # print(loss.sum(-1) / seg_num)

        loss = torch.sqrt((loss ** 2).reshape(batch_size, -1).sum(-1) / seg_num)
        return loss.sum()

    def seg_mask(self, seg_pred, threhold=None):
        return self.label(seg_pred, threhold=threhold)


    def Test_ori(self, model, loader, test_seg=False, half=False):
        
        print('testing model on NIST27 style ori ground truth [-90°-90°]')
        from tqdm import tqdm

        dev = self.dev

        ori_loss = 0
        seg_loss = 0
        imgs_number = 0

        for data in tqdm(loader):

            imgs, ori_truth, seg_truth, _ = data

            if dev is not None:
                imgs = imgs.to(dev)
                ori_truth = ori_truth.to(dev) 
                seg_truth = seg_truth.to(dev)
                # imgs = self.input_norm(imgs)

            if dev is not None:
                imgs = imgs.to(dev).half()
                ori_truth = ori_truth.to(dev)#.half() 
                seg_truth = seg_truth.to(dev)#.half()
                # imgs = self.input_norm(imgs)

            with torch.no_grad():

                ori, seg = model(imgs)
                # print(ori.dtype)
                ori, seg = self.downsample(ori, seg)
            
                # ------------------计算方向场----------------
                ori = ori * 180 / pi 
                ori_loss += self.test_ori_on_a_batch(-ori_truth, ori, seg_truth) #! WARNING: 标记的方向场标签与我们算的方向场标签时相反的标签
                imgs_number += imgs.shape[0]
                # -----------------计算 seg_IOU----------------
                if test_seg:
                    seg = self.seg_mask(seg, 0.5)
                    seg_loss += self.test_segIou_on_a_batch(seg_truth, seg)
                    
        loss = [ori_loss / imgs_number, seg_loss / imgs_number]
        print(f"the loss is {loss}, the number of imgs is {imgs_number}")
        return loss


    def test_segIou_on_a_batch(self, seg_true, seg_pred):

        batch_size = seg_true.shape[0]
        seg_true = seg_true.reshape(batch_size, -1)
        seg_pred = seg_pred.reshape(batch_size, -1)

        intersection = (seg_true * seg_pred).sum(-1)
        union = seg_true.sum(-1) + seg_pred.sum(-1) - intersection
        IoU = intersection.float() / union

        return IoU.sum()

    def input_norm(self, imgs):
        
        # mean = torch.mean(imgs, dim=(1,2,3), keepdim=True)
        # var = torch.var(imgs, dim=(1,2,3), keepdim=True)
        # imgs = (imgs - mean) / torch.sqrt(var)
        return imgs

    def gaussian_smooth(self, ori):

        if isinstance(ori, np.ndarray):
            ori = torch.tensor(ori)
        
        rad = self.get_rad(ori)
        sin_ori = torch.sin(rad * 2)
        cos_ori = torch.cos(rad * 2)

        angles = torch.stack((sin_ori, cos_ori), dim=1)
        
        angles = self.Guassian(angles)
        angles = self.Guassian(angles)
        # angles = self.Guassian(angles)

        angles = torch.atan2(angles[:, 0, ...], angles[:, 1, ...])

        return angles / 2

    
    def gaussian_kernel(self, shape=(5, 5), sigma=0.5):

        m, n = shape
        x = torch.arange(-(m-1) // 2, (m+1)// 2).view(-1, 1)
        y = torch.arange(-(n-1) // 2, (n+1)// 2)
        h = torch.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        kernel =  h / h.sum()

        return kernel



    # def mean_smooth(self, ori):
    
#     sin_res, cos_res = torch.sin(self.angle_kernel), torch.cos(self.angle_kernel)
#     sin_res = torch.sum(ori * sin_res, dim=1, keepdim=True)
#     cos_res = torch.sum(ori * cos_res, dim=1, keepdim=True)

#     sin_res = self.mean_conv2d(sin_res)
#     cos_res = self.mean_conv2d(cos_res)

#     return torch.atan2(sin_res, cos_res) / 2