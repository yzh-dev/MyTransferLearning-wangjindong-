# coding=utf-8
from torchvision import transforms
import numpy as np
import sklearn.model_selection as ms
import torch
from data.data_loader import DGImageTextData
from torch.utils.data import DataLoader

train_transfrom = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    # transforms.RandomGrayscale(),
    transforms.ToTensor(),  # 将通道转换为CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def get_img_dataloader(args, clip_preprocess=None):
    # ImageDataset(dataset=args.dataset, task=args.task, root_dir=args.data_dir, domain_name=names[i],
    #              domain_label=i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs)
    rate = 0.2
    trdatalist, tedatalist = [], []
    train_loaders, eval_loaders = [], []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)  # 计算domain的数量
    for i in range(len(names)):
        # 测试集所在domain
        if i in args.test_envs:
            tedatalist.append(DGImageTextData(dataset=args.dataset,
                                              rootdir=args.root,
                                              domain_name=names[i],
                                              preprocess=clip_preprocess,
                                              transform=test_transform,
                                              test_envs=args.test_envs,
                                              ))
        # 切分部分训练域中的数据，作为验证集
        else:
            tmpdatay = DGImageTextData(dataset=args.dataset,
                                       rootdir=args.root,
                                       domain_name=names[i],
                                       preprocess=clip_preprocess,
                                       transform=test_transform,
                                       test_envs=args.test_envs,
                                       ).int_labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(n_splits=2, test_size=rate, train_size=1 - rate,
                                                    random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                # 获取train domain中的train index和test index
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(DGImageTextData(dataset=args.dataset,
                                              rootdir=args.root,
                                              domain_name=names[i],
                                              preprocess=clip_preprocess,
                                              transform=test_transform,
                                              indices=indextr,
                                              test_envs=args.test_envs,
                                              ))
            tedatalist.append(DGImageTextData(dataset=args.dataset,
                                              rootdir=args.root,
                                              domain_name=names[i],
                                              preprocess=clip_preprocess,
                                              transform=test_transform,
                                              indices=indexte,
                                              test_envs=args.test_envs,
                                              ))

        train_loaders = [InfiniteDataLoader(dataset=env,
                                            weights=None,
                                            batch_size=args.batchsize,
                                            num_workers=args.N_WORKERS)
                         for env in trdatalist]

        eval_loaders = [DataLoader(dataset=env,
                                   batch_size=args.test_batchsize,
                                   num_workers=args.N_WORKERS,
                                   drop_last=False,
                                   shuffle=False,
                                   )
                        for env in trdatalist + tedatalist]  # 训练集加测试集

    return train_loaders, eval_loaders


# for mixup
def random_pairs_of_minibatches(minibatches):
    ld = len(minibatches)  # 领域个数
    pairs = []
    tdlist = np.arange(ld)
    batch_size = minibatches[0][0].size(0)
    txlist = np.arange(batch_size)
    for i in range(ld):
        for j in range(batch_size):
            # 关键：从ld个领域中随机选择两个不重复的领域(tdi, tdj)
            # 并从每个领域中随机选择两个样本(txi, txj)
            (tdi, tdj) = np.random.choice(tdlist, 2, replace=False)
            (txi, txj) = np.random.choice(txlist, 2, replace=True)
            if j == 0:  # 如果是该domain的第一个样本，则直接赋值
                imgi = torch.unsqueeze(minibatches[tdi][0][txi], dim=0)
                txti = torch.unsqueeze(minibatches[tdi][1][txi], dim=0)
                labeli = minibatches[tdi][2][txi]

                imgj = torch.unsqueeze(minibatches[tdj][0][txj], dim=0)
                txtj = torch.unsqueeze(minibatches[tdj][1][txj], dim=0)
                labelj = minibatches[tdj][2][txj]
            else:
                imgi = torch.vstack((imgi, torch.unsqueeze(minibatches[tdi][0][txi], dim=0)))
                txti = torch.vstack((txti, torch.unsqueeze(minibatches[tdi][1][txi], dim=0)))
                labeli = torch.hstack((labeli, minibatches[tdi][2][txi]))

                imgj = torch.vstack((imgj, torch.unsqueeze(minibatches[tdj][0][txj], dim=0)))
                txtj = torch.vstack((txtj, torch.unsqueeze(minibatches[tdj][1][txj], dim=0)))
                labelj = torch.hstack((labelj, minibatches[tdj][2][txj]))

        pairs.append(((imgi, txti, labeli), (imgj, txtj, labelj)))
    return pairs  # paris is a list ,length is ld


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:  # 死循环-->无限循环
            for batch in self.sampler:
                yield batch


# 可以无限载入数据
class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError
