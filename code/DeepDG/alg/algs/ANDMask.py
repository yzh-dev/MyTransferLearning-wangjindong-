# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
import torch.autograd as autograd


# _21 ICLR ANDMask_Learning explanations that are hard to vary.pdf
class ANDMask(ERM):
    def __init__(self, args):
        super(ANDMask, self).__init__(args)

        self.tau = args.tau

    def update(self, minibatches, opt, sch):

        total_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, data in enumerate(minibatches):  # 迭代domain
            x, y = data[0].cuda().float(), data[1].cuda().long()
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            # 当前domain下的loss作为env_loss
            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss
            # 计算当前domain下反传获取到的梯度env_grads
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)

        opt.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        opt.step()
        if sch:
            sch.step()

        return {'total': mean_loss.item()}

    # 只保留那些被更多数据认可的梯度方向
    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)  # num_domains,BCHW
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau  # BCHW
            mask = mask.to(torch.float32)  # BCHW
            avg_grad = torch.mean(grads, dim=0)  # 大小取不同domain下的平均梯度

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad  # 方向取更多domain认可的方向
            param.grad *= (1. / (1e-10 + mask_t))

        return 0
