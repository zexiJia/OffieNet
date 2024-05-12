import torch
from torch import nn
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class BaseNet(nn.Module):  
    def __init__(self, in_channels=1, out_channels=64):  
        super(BaseNet, self).__init__()  
          
        # 1-3：kernel is 11
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=1, padding=5)  # padding=(11-1)/2=5  
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=11, stride=1, padding=5)  # padding=(11-1)/2=5  
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm2d(out_channels)

          
        # 4-6：kernel is 9  
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=9, stride=1, padding=4)  # padding=(9-1)/2=4  
          
        # 7-9：kernel is 7
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)  # padding=(7-1)/2=3  

          
        # 9-12：kernel is 5
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)  # padding=(5-1)/2=2

        # 13：output
        self.conv5 = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)  # padding=(3-1)/2=1
          
    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(self.conv0(x))  
        x = self.relu(self.conv1(x))  
        x = self.relu(self.conv1(x))

        x = self.norm1(x)
        x = self.relu(self.conv2(x))  
        x = self.relu(self.conv2(x))  
        x = self.relu(self.conv2(x)) 

        x = self.norm1(x)
        x = self.relu(self.conv3(x))  
        x = self.relu(self.conv3(x))  
        x = self.relu(self.conv3(x)) 

        x = self.norm1(x)
        x = self.relu(self.conv4(x))  
        x = self.relu(self.conv4(x))  
        x = self.relu(self.conv4(x))  

        x = self.norm1(x)
        x = self.relu(self.conv5(x))  
        return x





    
