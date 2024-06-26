import torch
import torch.nn as nn


# https://zhuanlan.zhihu.com/p/163839117
# MMD的基本思想就是，如果两个随机变量的任意阶都相同的话，那么两个分布就是一致的。而当两个分布不相同的话，那么使得两个分布之间差距最大的那个矩应该被用来作为度量两个分布的标准
class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)  # 2B * num_feat
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))  # B * B * num_feat
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))  # B * B * num_feat
        L2_distance = ((total0-total1)**2).sum(2)  # 计算高斯核中的|x-y|,shape B * B
        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)  # 标量
        bandwidth /= kernel_mul ** (kernel_num // 2)  # 标量
        bandwidth_list = [bandwidth * (kernel_mul**i)  # list: kernel_num个items
                          for i in range(kernel_num)]
        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # 将多个核合并在一起

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])  # K_ss矩阵，Source<->Source
            YY = torch.mean(kernels[batch_size:, batch_size:])  # K_tt矩阵，Target<->Target
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
