# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


# _22 TMLR DIFEX_Domain-invariant Feature Exploration for Domain Generalization.pdf
# 分别让一半特征通过蒸馏学习域内不变特征，让一半特征通过对齐学习域间不变特征，并让特征之间不一致性尽量大，我们可以获得更多样的不变特征，最终获取更鲁棒的模型
# 域内不变特征（internally-invariant features），与分类有关的特征，产生于域的内部，不受其他域的影响，主要抓取数据的内在语义信息；
# 域间不变特征（mutually-invariant features），跨域迁移知识，通过多个域产生，共同学习的一些知识
# 已有的工作表明傅里叶相值（Phase）中包含更多的语义信息，不太容易受到域偏移的影响
# 参考https://zhuanlan.zhihu.com/p/546895864
class DIFEX(Algorithm):
    def __init__(self, args):
        super(DIFEX, self).__init__(args)
        self.args = args
        self.featurizer = get_fea(args)
        self.bottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)

        self.tfbd = args.bottleneck//2  # teaNet的瓶颈层只有stuNet的一半
        self.teaf = get_fea(args)
        self.teab = common_network.feat_bottleneck(self.featurizer.in_features, self.tfbd, args.layer)
        self.teac = common_network.feat_classifier(args.num_classes, self.tfbd, args.classifier)
        self.teaNet = nn.Sequential(self.teaf, self.teab, self.teac)

    # 训练teanet过程是基本的ERM过程
    # obtain internally-invariant features with Fourier phase information.
    def teanettrain(self, dataloaders, epochs, opt1, sch1):
        self.teaNet.train()
        minibatches_iterator = zip(*dataloaders)  # num_domain个InfiniteDataLoader
        for epoch in range(epochs):
            minibatches = [tdata for tdata in next(minibatches_iterator)]
            all_x = torch.cat([data[0].cuda().float() for data in minibatches])
            # 形状保持不变，关键
            all_z = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            all_y = torch.cat([data[1].cuda().long() for data in minibatches])
            all_p = self.teaNet(all_z)
            loss = F.cross_entropy(all_p, all_y, reduction='mean')
            opt1.zero_grad()
            loss.backward()

            if ((epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.7)) == 0 or (epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.9)) == 0) and (not self.args.schuse):
                for param_group in opt1.param_groups:
                    param_group['lr'] = param_group['lr']*0.1
            opt1.step()
            if sch1:
                sch1.step()

            if epoch % int(self.args.steps_per_epoch) == 0 or epoch == epochs-1:
                print('epoch: %d, cls loss: %.4f' % (epoch, loss))
        self.teaNet.eval()

    def coral(self, x, y):  # 对齐均值和方差
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        with torch.no_grad():
            all_x1 = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))  # 沿着dim=(2, 3)做变换，返回形状不变的张量
            tfea = self.teab(self.teaf(all_x1)).detach()

        all_z = self.bottleneck(self.featurizer(all_x))
        loss1 = F.cross_entropy(self.classifier(all_z), all_y)  # CE loss
        # guide the student network to learn the Fourier information.
        loss2 = F.mse_loss(all_z[:, :self.tfbd], tfea)*self.args.alpha  # internally-invariant features与tfea计算MSE loss
        if self.args.disttype == '2-norm':  # internally-invariant features与mutually-invariant features计算MSE loss
            loss3 = -F.mse_loss(all_z[:, :self.tfbd],
                                all_z[:, self.tfbd:])*self.args.beta
        elif self.args.disttype == 'norm-2-norm':
            loss3 = -F.mse_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                                all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
        elif self.args.disttype == 'norm-1-norm':
            loss3 = -F.l1_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                               all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
        elif self.args.disttype == 'cos':
            loss3 = torch.mean(F.cosine_similarity(all_z[:, :self.tfbd], all_z[:, self.tfbd:]))*self.args.beta
        loss4 = 0  # domain之间计算mutually-invariant features的Coral损失
        if len(minibatches) > 1:
            for i in range(len(minibatches)-1):
                for j in range(i+1, len(minibatches)):
                    loss4 += self.coral(all_z[i*self.args.batch_size:(i+1)*self.args.batch_size, self.tfbd:],
                                        all_z[j*self.args.batch_size:(j+1)*self.args.batch_size, self.tfbd:])
            loss4 = loss4*2/(len(minibatches) *
                             (len(minibatches)-1))*self.args.lam
        else:
            loss4 = self.coral(all_z[:self.args.batch_size//2, self.tfbd:],
                               all_z[self.args.batch_size//2:, self.tfbd:])
            loss4 = loss4*self.args.lam

        loss = loss1+loss2+loss3+loss4
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss1.item(), 'dist': (loss2).item(), 'exp': (loss3).item(), 'align': loss4.item(), 'total': loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))
