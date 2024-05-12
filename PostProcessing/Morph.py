import torch

class Dilate(torch.nn.Module):
    def __init__(self, r, cpu=False):
        super().__init__()
        self.r = r
        self.weight = torch.ones(1, 1, r, r)
        if not cpu:
            self.weight = self.weight.cuda()
    
    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=int(self.r // 2))
        return torch.where(x > 0, 1, 0)


class Erode(torch.nn.Module):
    def __init__(self, r, cpu=False):
        super().__init__()
        self.r = r
        self.weight = torch.ones(1, 1, r, r)
        if not cpu:
            self.weight = self.weight.cuda()
        self.target = self.r ** 2
    
    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=int(self.r // 2))
        return torch.where(x == self.target, 1, 0)

   