import torch
from torch.nn import Module, Conv2d, InstanceNorm2d, ReLU, ConvTranspose2d, PReLU, Softmax, Sigmoid, Tanh
import torch.nn.functional as F
from numpy import pi, array
import logging
from FingerNet.blocks import CIP, res_block



class Orientation(torch.nn.Module):
    '''
        对比与以前的方向场，进行了加宽，差异在于将原先的softmax 180维度向量的输出改为cos2θ，sin2θ的输出了
        输出的地方的key改为：
        self.ori_unet_res_output = Conv2d(96, 2, (1, 1))
        self.ori_res_output = Tanh()
    '''

    def __init__(self, Temp=1):
        super(Orientation, self).__init__()

        self.Temp = Temp
        self.relu = ReLU(True)
        self.sigmoid = Sigmoid()

        # self.inputPad = torch.nn.ZeroPad2d((2, 3, 2, 3))
        self.conv1 = Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3))
        self.InsN1 = InstanceNorm2d(64, affine=True)

        self.block_down1 = res_block(64, [64, 64, 128], kernel_size=(3, 3), dilation=(1, 1), stride=(2, 2))
        self.block1 = res_block(128, [64, 64, 128], kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1))

        self.OF_down2 = res_block(128, [96, 96, 192], kernel_size=(3, 3), dilation=(1, 1), stride=(2, 2))
        self.OF = res_block(192, [96, 96, 192], kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1))

        self.seg_pre = res_block(192, [128, 128, 256], kernel_size=(3, 3), dilation=(2, 2), stride=(1, 1), down=True) #并不下采样
        self.seg_middle = res_block(256, [128, 128, 256], kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1))
        self.seg = res_block(256, [128, 128, 256], kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1))

        # OF --> ori
        self.ori_unet_down1 = res_block(192, [128, 128, 256], kernel_size=(3, 3),dilation=(1, 1), stride=(2,2))
        self.ori_unet1 = res_block(256, [128, 128, 256], kernel_size=(3, 3),dilation=(1, 1), stride=(1,1))

        self.ori_unet_down2 = res_block(256, [256, 256, 512], kernel_size=(3, 3),dilation=(1, 1), stride=(2,2))
        self.ori_unet2 = res_block(512, [256, 256, 512], kernel_size=(3, 3),dilation=(1, 1), stride=(1,1))

        self.ori_unet_trans1 = ConvTranspose2d(512, 256, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.ori_trans_ins1 = InstanceNorm2d(256, affine=True)

        self.ori_unet_middle = res_block(256+256, [128, 128, 256], kernel_size=(3, 3),dilation=(1, 1), stride=(1,1), down=True)

        self.ori_unet_trans2 = ConvTranspose2d(256, 192, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.ori_trans_ins2 = InstanceNorm2d(192, affine=True)

        self.ori_unet_output_pre = res_block(192+192, [64, 64, 128], kernel_size=(3, 3),dilation=(1, 1), stride=(1,1), down=True)
        self.ori_unet_output_middle = CIP(128, 128, kernel_size=(1, 1),padding=0, dilation=(1, 1)) # conv+Instance+PReLU

        self.ori_unet_output = Conv2d(128, 2, (1, 1))
        self.ori_output = Tanh()



        #seg -> seg_ouput
        self.seg_1 = CIP(256, 256, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1))
        self.seg_2 = CIP(256, 256, kernel_size=(3, 3), padding=(4, 4), dilation=(4, 4))
        self.seg_3 = CIP(256, 256, kernel_size=(3, 3), padding=(8, 8), dilation=(8, 8))
        self.seg_4 = CIP(256, 256, kernel_size=(1, 1), padding=(0, 0), dilation=(1, 1))

        self.seg_5 = CIP(256*4, 256, kernel_size=(1, 1), padding=(0, 0), dilation=(1, 1))
        self.seg_output = Conv2d(256, 1, (1, 1))


    def forward(self, x):

        # x = self.inputPad(x)

        # Texture
        texture = self.getTexture(x)

        # OF
        OF = self.getOF(texture)

        #QF
        QF = self.getQF(OF)

        # Unet of ori
        output = self.getOrientation(OF)

        # QF
        seg_res = self.getSegment(QF)

        return output, seg_res, texture,  OF # 代表不加 guassian, 加了 gaussian 的方向 、分割

    def getTexture(self, x):
        
        x = self.conv1(x)
        x = self.InsN1(x)
        x = self.relu(x)

        x = self.block_down1(x)
        texture = self.block1(x)

        return texture


    def getOF(self, texture):


        # OF
        x = self.OF_down2(texture)
        OF = self.OF(x)

        return OF
    

    def getQF(self, OF):

        QF = self.seg_pre(OF)
        QF = self.seg_middle(QF)
        QF = self.seg(QF)

        return QF

    def getOrientation(self, OF):
                # Unet of ori
        ori = self.ori_unet_down1(OF)
        sub2 = self.ori_unet1(ori)
        ori = self.ori_unet_down2(sub2)
        ori = self.ori_unet2(ori)

        ori = self.ori_unet_trans1(ori)
        ori = self.ori_trans_ins1(ori)
        ori = torch.cat((ori, sub2), 1)# 链接
        ori = self.ori_unet_middle(ori)

        ori = self.ori_unet_trans2(ori)
        ori = self.ori_trans_ins2(ori)
        ori = torch.cat((ori, OF), 1) # 链接

        ori = self.ori_unet_output_pre(ori)
        ori = self.ori_unet_output_middle(ori)
        ori = self.ori_unet_output(ori)
        
        output = self.ori_output(ori)
        
        return output


    def getSegment(self, QF):

        seg1 = self.seg_1(QF)
        seg2 = self.seg_2(seg1)
        seg3 = self.seg_3(seg2)
        seg4 = self.seg_4(seg3)

        seg_all = torch.cat((seg1, seg2, seg3, seg4), 1)
        seg5 = self.seg_5(seg_all)
        seg_res = self.seg_output(seg5)
        seg_res = self.sigmoid(seg_res)

        return seg_res


    def freezeOrientantionMoudle(self):

        self.conv1.requires_grad_(False)
        self.InsN1.requires_grad_(False)

        self.block_down1.requires_grad_(False)
        self.block1.requires_grad_(False)

        self.OF_down2.requires_grad_(False)
        self.OF.requires_grad_(False)

        # OF --> ori
        # self.ori_unet_down1.requires_grad_(False)
        # self.ori_unet1.requires_grad_(False)

        # self.ori_unet_down2.requires_grad_(False)
        # self.ori_unet2.requires_grad_(False)

        # self.ori_unet_trans1.requires_grad_(False)
        # self.ori_trans_ins1.requires_grad_(False)

        # self.ori_unet_middle.requires_grad_(False)

        # self.ori_unet_trans2.requires_grad_(False)
        # self.ori_trans_ins2.requires_grad_(False)

        # self.ori_unet_output_pre.requires_grad_(False)
        # self.ori_unet_output_middle.requires_grad_(False)

        # self.ori_unet_output.requires_grad_(False)

        self.seg_pre.requires_grad_(False)
        self.seg_middle.requires_grad_(False)
        self.seg.requires_grad_(False)
        self.seg_1.requires_grad_(False)
        self.seg_2.requires_grad_(False)
        self.seg_3.requires_grad_(False)
        self.seg_4.requires_grad_(False)
        self.seg_5.requires_grad_(False)
        self.seg_output.requires_grad_(False)




