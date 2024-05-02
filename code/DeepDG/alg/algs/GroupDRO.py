# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst domain
    DISTRIBUTIONALLY ROBUST NEURAL NETWORKS FOR GROUP SHIFTS: ON THE IMPORTANCE OF REGULARIZATION FOR WORST-CASE GENERALIZATION
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, args):
        super(GroupDRO, self).__init__(args)
        self.register_buffer("q", torch.Tensor())
        self.args = args

    def update(self, minibatches, opt, sch):

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).cuda()

        losses = torch.zeros(len(minibatches)).cuda()

        for m in range(len(minibatches)):  # 迭代不同domain
            x, y = minibatches[m][0].cuda().float(), minibatches[m][1].cuda().long()
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.args.groupdro_eta * losses[m].data).exp()  # 相当于是对当前domain下的loss进行加权

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {'group': loss.item()}
