# coding=utf-8
import torch
import torch.nn.functional as F

from alg.algs.ERM import ERM


class MMD(ERM):
    def __init__(self, args):
        super(MMD, self).__init__(args)
        self.args = args
        self.kernel_type = "gaussian"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):  # x: batch * feat_dim
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, opt, sch):
        clf_loss = 0
        mmd_loss = 0
        num_domains = len(minibatches)

        features = [self.featurizer(
            data[0].cuda().float()) for data in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [data[1].cuda().long() for data in minibatches]
        # 計算domains之間的MMD損失作爲正則化項
        for i in range(num_domains):
            clf_loss += F.cross_entropy(classifs[i], targets[i])  # clf loss
            for j in range(i + 1, num_domains):
                mmd_loss += self.mmd(features[i], features[j])  # loss between domains

        clf_loss /= num_domains
        if num_domains > 1:
            mmd_loss /= (num_domains * (num_domains - 1) / 2)
        total_loss = clf_loss + self.args.mmd_gamma*mmd_loss
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {'class': clf_loss.item(), 'mmd': mmd_loss.item(), 'total': total_loss.item()}
