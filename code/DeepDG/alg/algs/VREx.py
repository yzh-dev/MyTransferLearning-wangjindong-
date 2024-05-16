# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


# _21 ICML VREx_Out-of-Distribution Generalization via Risk Extrapolation (REx).pdf
class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, args):
        super(VREx, self).__init__(args)
        self.register_buffer('update_count', torch.tensor([0]))  # 初始化为0
        self.args = args

    def update(self, minibatches, opt, sch):
        if self.update_count >= self.args.anneal_iters:
            penalty_weight = self.args.lam
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))

        for i, data in enumerate(minibatches):  # 迭代不同domain
            logits = all_logits[all_logits_idx:all_logits_idx + data[0].shape[0]]
            all_logits_idx += data[0].shape[0]
            nll = F.cross_entropy(logits, data[1].cuda().long())
            losses[i] = nll  # 计算该domain下的loss

        mean = losses.mean()  # mean loss of different domains
        penalty = ((losses - mean) ** 2).mean()  # 等价于将不同domain下loss的方差作为惩罚系数，强制模型在不同domain下损失尽量相同
        loss = mean + penalty_weight * penalty

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        self.update_count += 1
        return {'total': loss.item(), 'domains_mean': mean.item(),
                'penalty': penalty.item()}
