from FingerNet.blocks import res_block
from torch import nn
import torch

class CLS(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = res_block(256, [128, 128, 256], (3, 3))
        self.conv2 = res_block(256, [128, 128, 256], (3, 3), stride=(1, 1))
        self.conv3 = res_block(256, [128, 128, 256], (3, 3))
        self.conv4 = res_block(256, [128, 128, 256], (3, 3), stride=(1, 1))        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(256, 64, True)
        self.prule = nn.PReLU(64, 0)
        self.out = nn.Linear(64, 1)
        self.sigmod = nn.Sigmoid()


    def forward(self, QF):
        x = self.conv1(QF)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = x.reshape(x.size(0), x.size(1))
        x = self.dense(x)
        x = self.prule(x)
        x = self.out(x)
        x = self.sigmod(x)
        return x

