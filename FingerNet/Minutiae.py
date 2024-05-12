import torch
from FingerNet.blocks import res_block, CIP
from torch.nn import Conv2d, InstanceNorm2d, ReLU, Sigmoid, Softmax, BatchNorm2d, MaxPool2d
import logging
class Minutiae(torch.nn.Module):

    def __init__(self):
        '''
            重新写过的 细节点模块原因在于先前的细节点模块少些了几个卷积
        '''

        super(Minutiae, self).__init__()
        # logging.getLogger('root').info('sigmod 注意')
        self.relu = ReLU(inplace=True)
        self.sigmoid = Sigmoid()
        self.tanh = torch.nn.Tanh()

        self.enhence_conv1 = Conv2d(1, 64, 7, 2, 3)
        self.enhence_bn = BatchNorm2d(64)
        self.enhence_block2_down = res_block(in_channel=64, out_channels=[64, 64, 128], kernel_size=(3, 3))
        self.enhence_block2 = res_block(in_channel=128, out_channels=[64, 64, 128], kernel_size=(3, 3), stride=(1, 1))
        self.enhence_block3_down = res_block(in_channel=128, out_channels=[128, 128, 256], kernel_size=(3, 3))
        self.enhence_block3 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))

        self.texture_block1_down = res_block(in_channel=128, out_channels=[96, 96, 192], kernel_size=(3, 3))
        self.texture_block1 = res_block(in_channel=192, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), down=True)
        # self.texture_block2 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))

        self.OF_block1 = res_block(in_channel=192, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1), down=True)
        self.OF_block2 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))
        self.OF_block3 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))

        self.fusion_1 = res_block(in_channel=256 * 3, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1), down=True)
        self.fusion_2 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))
        self.fusion_3 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))
        self.fusion_4 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))
        self.fusion_5 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))
        self.fusion_6 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))
        self.fusion_7 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2),down=True)
        self.fusion_8 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))
        self.fusion_9 = res_block(in_channel=256, out_channels=[128, 128, 256], kernel_size=(3, 3), stride=(1, 1))

        self.aspp_scale_1 = res_block(256, [128, 128, 256], kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1))
        self.aspp_scale_2 = res_block(256, [128, 128, 256], kernel_size=(3, 3), dilation=(4, 4), stride=(1, 1))
        self.aspp_scale_3 = res_block(256, [128, 128, 256], kernel_size=(3, 3), dilation=(8, 8), stride=(1, 1))
        self.aspp_fusion = res_block(256, [128, 128, 256], kernel_size=(3, 3), dilation=(1, 1), stride=(1, 1))

        self.output_ori_1 = CIP(2+256, 256, 1, 0)
        self.output_ori_2 = Conv2d(256, 2, 1)

        self.output_w_1 = CIP(256, 256, 1, 0)
        self.output_w_2 = Conv2d(256, 1, 1)

        self.output_h_1 = CIP(256, 256, 1, 0)
        self.output_h_2 = Conv2d(256, 1, 1)

        self.output_confidence_1 = CIP(256, 256, 1, 0)
        self.output_confidence_2 = Conv2d(256, 1, 1)

    
    def forward(self, enhence_image, texture, OF, orientation, segment):
        
        fusion_feature = self.get_fusion_feature(enhence_image, texture, OF, segment)
        aspp_feature = self.get_ASPP(fusion_feature)

        confidence = self.get_confidence(aspp_feature)
        w, h = self.get_position(aspp_feature)
        orientation = self.get_mnt_ori(aspp_feature, orientation)

        return confidence, w, h, orientation


    def get_fusion_feature(self, enhence_image, texture, OF, segment):

        enhence = self.enhence_conv1(enhence_image)
        enhence = self.enhence_bn(enhence)
        enhence = self.relu(enhence)
        enhence = self.enhence_block2_down(enhence)
        enhence = self.enhence_block2(enhence)
        enhence = self.enhence_block3_down(enhence)
        enhence = self.enhence_block3(enhence)

        texture = self.texture_block1_down(texture)
        texture = self.texture_block1(texture)
        # texture = self.texture_block2(texture)

        OF = self.OF_block1(OF)
        OF = self.OF_block2(OF)
        OF = self.OF_block3(OF)

        fusion = torch.cat((texture, OF, enhence), dim=1)
        fusion = fusion * segment
        fusion = self.fusion_1(fusion)
        fusion = self.fusion_2(fusion)
        fusion = self.fusion_3(fusion)
        fusion = self.fusion_4(fusion)
        fusion = self.fusion_5(fusion)
        fusion = self.fusion_6(fusion)
        fusion = self.fusion_7(fusion)
        fusion = self.fusion_8(fusion)
        fusion = self.fusion_9(fusion)

        return fusion

    
    def get_ASPP(self, fusion):

        level_1_feature = self.aspp_scale_1(fusion)
        level_2_feature = self.aspp_scale_2(fusion)
        level_3_feature = self.aspp_scale_3(fusion)

        aspp_feature = level_1_feature + level_2_feature + level_3_feature
        aspp_feature = self.aspp_fusion(aspp_feature)

        return aspp_feature


    def get_mnt_ori(self, aspp_feature, orientation):

        mnt_ori_feature = torch.cat((aspp_feature, orientation), dim=1)
        mnt_ori = self.output_ori_1(mnt_ori_feature)
        mnt_ori = self.output_ori_2(mnt_ori)
        mnt_ori = self.tanh(mnt_ori)

        return mnt_ori

    
    def get_position(self, aspp_feature):

        mnt_w = self.output_w_1(aspp_feature)
        mnt_w = self.output_w_2(mnt_w)
        mnt_w = self.sigmoid(mnt_w)

        mnt_h = self.output_h_1(aspp_feature)
        mnt_h = self.output_h_2(mnt_h)
        mnt_h = self.sigmoid(mnt_h)

        return mnt_w, mnt_h

    
    def get_confidence(self, aspp_feature):

        mnt_confidence = self.output_confidence_1(aspp_feature)
        mnt_confidence = self.output_confidence_2(mnt_confidence)
        mnt_confidence = self.sigmoid(mnt_confidence)

        return mnt_confidence


