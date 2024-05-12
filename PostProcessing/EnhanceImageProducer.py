import sys
sys.path.append('..')
from PostProcessing.OrientationTools import OrientationTools
from PostProcessing.SegmentLabelTools import SegmentLabelTools
from scipy import misc

import torch

pi = 3.141592653589793

class EnhanceImageProducer(torch.nn.Module):

    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride
        rad = torch.arange(-90, 90, self.stride) * pi / 180
        print(rad.shape)
        self.kernel = self.gabor_function(-rad)[0]
        # self.kernel = torch.tensor(gabor_bank()[0]).reshape(90,1, 25, 25)
        self.gabor_conv = torch.nn.Conv2d(1, 180, 25, 1, padding=12, bias=False)
        self.gabor_conv.weight = torch.nn.Parameter(self.kernel)
        self.gabor_conv.weight.requires_grad = False
        self.max_angle = 180 // self.stride


    def gabor_function(self, theta, ksize=25, sigma=4.5, Lambda=8, psi=0, gamma=0.5):
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        elements = torch.arange(-(ksize // 2), (ksize // 2 + ksize % 2)).reshape(-1, ksize)
        y = (torch.ones((ksize, ksize)) * elements).reshape((1, 1, ksize, ksize))
        x = (torch.ones((ksize, ksize)) * elements.T).reshape((1, 1, ksize, ksize))
        theta = theta.reshape(-1, 1, 1, 1)
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
        gb_cos = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * pi / Lambda * x_theta + psi)
        gb_sin = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.sin(2 * pi / Lambda * x_theta + psi)
        return gb_cos,  gb_sin

    @torch.no_grad()
    def forward(self, imgs, oris): 
        res = self.gabor_conv(imgs)
        oris = torch.nn.functional.interpolate(oris.float(), scale_factor=8,recompute_scale_factor=True ,mode='bilinear')
        angle = (torch.atan2(oris[:, 0:1, ...], oris[:, 1:, ...]) * 90 / pi + 90) / self.stride
        res = torch.gather(res, 1, angle.long() % self.max_angle)
        enhence = torch.sum( res, dim=1, keepdim=True)
        # enhence = torch.where(enhence>0, 1., -1.)
        return enhence.detach()



    

