import torch 
import torch.nn as nn
import torchvision
import numpy as np


class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class Deformconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 
 
        self.conv_offset = nn.Conv2d(in_ch, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, in_ch, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset, requires_grad=True)
 
        self.conv_mask = nn.Conv2d(in_ch, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, in_ch, 3, 3])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask, requires_grad=True)
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                            bias=self.conv.bias,
                                            padding=(1, 1),
                                            mask=mask)
        return out
    

class ScaleAwareDeformconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ScaleAwareDeformconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # 2D bboxes size encoding
        self.glob_offset = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        # Feature map encoding
        self.local_offset = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, padding=0)
        )

        self.fuse_offset = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * 3 * 2, 1, padding=0)
        )

        self.conv_mask_prepro = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv_mask = nn.Conv2d(in_ch, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, in_ch, 3, 3])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask, requires_grad=True)
        self.simam = simam_module()

    def forward(self, x, scale_box2d):
        bsize = torch.stack((scale_box2d[:, 4] - scale_box2d[:, 2], scale_box2d[:, 3] - scale_box2d[:, 1]), dim=-1)
        b, c, h, w = x.shape
        g_offset = self.glob_offset(bsize)
        g_offset = g_offset.view(b, -1, 1, 1).repeat(1, 1, h, w).contiguous()
        l_offset = self.local_offset(x)
        offset = self.fuse_offset(torch.cat((g_offset, l_offset), dim=1))

        x0 = self.simam(self.act(self.conv_mask_prepro(x)))
        mask = torch.sigmoid(self.conv_mask(x0))
        # mask = torch.sigmoid(self.conv_mask(x))
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                            bias=self.conv.bias,
                                            padding=(1, 1),
                                            mask=mask)
        return out
    

class DepthScaleAwareDeformconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthScaleAwareDeformconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # 2D bboxes size encoding
        self.glob_offset = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        # Feature map encoding
        self.local_offset = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, padding=0)
        )

        self.fuse_offset = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * 3 * 2, 1, padding=0)
        )

        self.conv_mask = nn.Conv2d(in_ch, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, in_ch, 3, 3])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask, requires_grad=True)

    def forward(self, x, scale_box2d):
        bsize = torch.stack((scale_box2d[:, 4] - scale_box2d[:, 2], scale_box2d[:, 3] - scale_box2d[:, 1]), dim=-1)
        b, c, h, w = x.shape
        g_offset = self.glob_offset(bsize)
        g_offset = g_offset.view(b, -1, 1, 1).repeat(1, 1, h, w).contiguous()
        l_offset = self.local_offset(x)
        offset = self.fuse_offset(torch.cat((g_offset, l_offset), dim=1))

        mask = torch.sigmoid(self.conv_mask(x))
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                            bias=self.conv.bias,
                                            padding=(1, 1),
                                            mask=mask)
        return out
    
    
if __name__ == "__main__":
    a = torch.zeros([8, 256, 64, 120])
    c = Deformconv(256, 2)
    res = c(a)
    print(res.shape)