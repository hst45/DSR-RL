import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class _NonLocalBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=1,
                 sub_sample=False,
                 bn_layer=True):
        """
        :param in_channels:输入通道数，论文中是1024
        :param inter_channels:内部通道数，论文中是512
        :param dimension:1维是向量，2维是图片，3维是视频
        :param sub_sample:是否需要对输入下采样
        :param bn_layer:是否使用batch normalization
        """

        super(_NonLocalBlockND, self).__init__ ()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 如果没有初始化内部通道数，默认为输入通道的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 如果处理视频就用3D卷积，另外两个类似
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size = (1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size = (2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size = (2))
            bn = nn.BatchNorm1d

        # g(·), 输入的线性嵌入
        # 1*1（或1*1*1）卷积改变输入x的通道数为输入通道数的一半
        self.g = conv_nd(in_channels = self.in_channels,
                          out_channels = self.inter_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0)

        # 同样用1*1（或1*1*1）卷积将输出通道数放大一倍，保持和输入通道数相同
        # 根据bn_layer参数决定是否在卷积后加一个batch normalization
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels = self.inter_channels,
                        out_channels = self.in_channels,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels = self.inter_channels,
                              out_channels = self.in_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # 两个嵌入函数，作用和g(·)一样
        self.theta = conv_nd(in_channels = self.in_channels,
                              out_channels = self.inter_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0)
        self.phi = conv_nd(in_channels = self.in_channels,
                            out_channels = self.inter_channels,
                            kernel_size = 1,
                            stride = 1,
                            padding = 0)

        # 根据sub_sample决定是否需要下采样
        # 注意到此处并没有对theta的输出下采样
        # 否则theta和phi的输出相乘后大小和g的输出不一样
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=True):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map:
            if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)

        # 把g的输出铺平为(bs, H*W, C/2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 把theta和phi的输出同样铺平为(bs, H*W, C/2)
        # 矩阵相乘，在用Softmax得到两个位置之间的相关性
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # phi_x = phi_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim = -1)

        # 位置相关性 和 输入的线性嵌入 相乘
        y = torch.matmul(f_div_C, g_x)
        # resize为(bs, H, W, c/2)
        # contiguous：如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的
        #             如果Tensor是连续的，则contiguous无操作。
        # y = y.permute(0, 2, 1).contiguous()
        y = y.contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # 经过W变回输入大小(bs, H, W, c)
        W_y = self.W(y)
        # 最终的输出z
        # z = W_y + x

        if return_nl_map:
            return W_y, f_div_C
        return W_y


class _NonLocalBlock2D(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=False,
                 bn_layer=True):
        """
        :param in_channels:输入通道数，论文中是1024
        :param inter_channels:内部通道数，论文中是512
        :param dimension:1维是向量，2维是图片，3维是视频
        :param sub_sample:是否需要对输入下采样
        :param bn_layer:是否使用batch normalization
        """

        super(_NonLocalBlock2D, self).__init__ ()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 如果没有初始化内部通道数，默认为输入通道的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 如果处理视频就用3D卷积，另外两个类似
        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size = (2, 2))
        bn = nn.BatchNorm2d

        # g(·), 输入的线性嵌入
        # 1*1（或1*1*1）卷积改变输入x的通道数为输入通道数的一半
        self.g = conv_nd(in_channels = self.in_channels,
                          out_channels = self.inter_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0)

        # 同样用1*1（或1*1*1）卷积将输出通道数放大一倍，保持和输入通道数相同
        # 根据bn_layer参数决定是否在卷积后加一个batch normalization
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels = self.inter_channels,
                        out_channels = self.in_channels,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels = self.inter_channels,
                              out_channels = self.in_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # 两个嵌入函数，作用和g(·)一样
        self.theta = conv_nd(in_channels = self.in_channels,
                              out_channels = self.inter_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0)
        self.phi = conv_nd(in_channels = self.in_channels,
                            out_channels = self.inter_channels,
                            kernel_size = 1,
                            stride = 1,
                            padding = 0)

        # 根据sub_sample决定是否需要下采样
        # 注意到此处并没有对theta的输出下采样
        # 否则theta和phi的输出相乘后大小和g的输出不一样
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=True):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map:
            if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)

        # 把g的输出铺平为(bs, H*W, C/2)
        g_x = self.g(x).permute(0, 1, 3, 2)

        # 把theta和phi的输出同样铺平为(bs, H*W, C/2)
        # 矩阵相乘，在用Softmax得到两个位置之间的相关性
        theta_x = self.theta(x).permute(0, 1, 3, 2)
        phi_x = self.phi(x)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim = -1)

        # 位置相关性 和 输入的线性嵌入 相乘
        y = torch.matmul(f_div_C, g_x)
        # resize为(bs, H, W, c/2)
        # contiguous：如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的
        #             如果Tensor是连续的，则contiguous无操作。
        y = y.permute(0, 1, 3, 2).contiguous()
        # 经过W变回输入大小(bs, H, W, c)
        W_y = self.W(y)
        # 最终的输出z
        z = W_y + x

        if return_nl_map:
            return W_y, f_div_C
        return W_y