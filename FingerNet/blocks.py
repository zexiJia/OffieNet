import torch
from torch.nn import Module, Conv2d, InstanceNorm2d, ReLU, ConvTranspose2d, PReLU, Softmax, Sigmoid, ReLU6, Sequential

'''
经过检查发现，有些地方和原始的 FingerNet 的实现还是有些不一样的地方，所以有 blocks 更改过来

'''

class res_block(Module):

    def __init__(self, in_channel:int, out_channels:list, kernel_size:tuple, dilation:tuple=(1, 1), stride:tuple=(2, 2), down:bool=False, affine:bool=True):
        
        super(res_block, self).__init__()

        assert stride[0] <= 2, print('stride must less than 2')
        oc1, oc2, oc3 = out_channels
        self.down = (stride[0] > 1) or down # 如果 stride > 1 则必须要下采样，或者指定 down=True 强制加上 conv4
        self.relu = ReLU(inplace=True)

        self.conv1 = Conv2d(in_channel, oc1, (1, 1), stride=stride)
        self.InsN1 = InstanceNorm2d(oc1, affine=affine)

        p = (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)) // 2
        self.conv2 = Conv2d(oc1, oc2, kernel_size, dilation=dilation, padding=p)
        self.InsN2 = InstanceNorm2d(oc2, affine=affine)

        self.conv3 = Conv2d(oc2, oc3, (1, 1))
        self.InsN3 = InstanceNorm2d(oc3, affine=affine)

        if self.down:
            self.conv4 = Conv2d(in_channel, oc3, (1, 1), stride=stride)
            self.InsN4 = InstanceNorm2d(oc3,affine=affine)

        
    def forward(self, x):

        inp = x
        x = self.conv1(x)
        x = self.InsN1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.InsN2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.InsN3(x)

        if self.down: 
            inp = self.conv4(inp)
            inp = self.InsN4(inp)

        return self.relu(x+inp)


class CIP(Module):

    def __init__(self, in_channel, out_channels, kernel_size=(3, 3), padding=1, dilation=(1, 1)):
        super(CIP, self).__init__()
        self.conv = Conv2d(in_channel, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.ins = InstanceNorm2d(out_channels, affine=True)
        self.active = PReLU(out_channels, 0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.ins(x)
        x = self.active(x)
        return x



class detph_block(Module):

    def __init__(self, input_channel, input_expansion, kernel_size, stride, output_channel, add_final=False):
        
        super(detph_block, self).__init__()

        self.add_final = (add_final and stride == 1 and input_channel == output_channel)
        self.relu = ReLU6(True)

        expansion_channel = int(input_channel * input_expansion)
        self.conv1 = Conv2d(input_channel, expansion_channel, 1)
        self.ins1 = InstanceNorm2d(expansion_channel, affine=True)

        padding = kernel_size // 2
        self.detph_wise = Conv2d(expansion_channel, expansion_channel, kernel_size, stride, padding, groups=expansion_channel)
        self.ins2 = InstanceNorm2d(expansion_channel, affine=True)

        self.conv2 = Conv2d(expansion_channel, output_channel, 1, 1)
        self.ins3 = InstanceNorm2d(output_channel, affine=True)

    def forward(self, inp):

        x = self.conv1(inp)
        x = self.ins1(x)
        x = self.relu(x)

        x = self.detph_wise(x)
        x = self.ins2(x)
        x  = self.relu(x)

        x = self.conv2(x)
        x = self.ins3(x)

        if self.add_final :
            return x + inp

        return x


class DetphWiseBlocks(Module):

    def __init__(self, in_channel:int, input_expansion:int, out_channels:int, kernel_size:int, dilation:int=1, stride:int=2, number_of_block=1):
        
        super(DetphWiseBlocks, self).__init__()

        pipe_line = []
        pipe_line.append(detph_block(in_channel, input_expansion, kernel_size, stride, out_channels))

        if number_of_block > 1:
            other_layers = [detph_block(out_channels, input_expansion, kernel_size, 1, out_channels) for i in range(number_of_block - 1)]
            pipe_line += other_layers

        self.layers = Sequential(*pipe_line)

    def forward(self, x):
        return self.layers(x)



