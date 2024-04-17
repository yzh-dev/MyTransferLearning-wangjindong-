# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class ERM(object):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(object, self).__init__(args)

    def update(self, minibatches, opt, sch):# minibatches:domain_nums个list
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)
