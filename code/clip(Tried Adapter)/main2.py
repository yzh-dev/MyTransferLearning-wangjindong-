import argparse
import numpy as np
import os

import pandas as pd
import pretty_errors
import torch
import torch.optim as optim
import torchvision
from sklearn.linear_model import LogisticRegression
from CustomClip import ClipModel
from data.data_loader import ImageTextData
from data.getdataloader import get_img_dataloader, train_valid_target_eval_names
from utils import gather_res, get_logger, set_gpu, set_seed
import wandb
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import sklearn.model_selection as ms

# RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
torch.backends.cudnn.benchmark = True
import pytorch_warmup as warmup


# torch.autograd.set_detect_anomaly(True, False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['zs', 'fe', 'ft', 'fe_lp', 'lp', 'fe_byResNet', 'ftAdapter'],
                        default='ftAdapter')
    # parser.add_argument('--dataset', type=int, default=0)  # 训练域数据
    parser.add_argument('--dataset', type=str, default='OfficeHome')  # 训练域数据
    parser.add_argument('--traindomain', type=int, default=0)  # 训练域数据
    parser.add_argument('--model', type=int, default=0)  # -1 for sweep
    parser.add_argument('--root', type=str, default='../../data/dataset/')  # root path of dataset
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--seed', type=int, default=42)  # random seed
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--result', action='store_true')  # if you want to sweep results statistics
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--N_WORKERS', type=int, default=2)
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--cliplr', type=float, default=1e-5)
    parser.add_argument('--basiclr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='DANN GRL alpha')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--schuse', action='store_true', default=True)
    parser.add_argument('--mixupuse', action='store_true')
    parser.add_argument('--dduse', action='store_true', help='use domain discriminator')
    parser.add_argument('--coraluse', action='store_true')
    parser.add_argument('--ImgAdapUse', action='store_true')
    parser.add_argument('--TxtAdapUse', action='store_true')
    parser.add_argument('--WarmUpUse', action='store_true', default=True)
    parser.add_argument('--ContrastLossUse', action='store_true', default=True)#对比损失
    parser.add_argument('--ClipMode', type=str, default='Frozen', choices=["Frozen", "Partial", "Full"])  # Clip模式选择
    parser.add_argument('--test_batchsize', type=int, default=16)
    parser.add_argument('--test_data', type=int, default=1)  # 测试域数据
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0], help='target domains')
    args = parser.parse_args()
    # 补充部分参数
    args = img_param_init(args)
    print(args)
    return args


# 参数初始化
def img_param_init(args):
    dataset = args.dataset
    args.img_dataset = {
        'office31': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
    }

    if dataset == 'office31':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'OfficeHome':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    else:
        print('No such dataset exists!')
    args.domains = domains

    if args.dataset == 'OfficeHome':
        args.num_classes = 65
    elif args.dataset == 'office31':
        args.num_classes = 31
    elif args.dataset == 'PACS':
        args.num_classes = 7
    elif args.dataset == 'VLCS':
        args.num_classes = 5

    return args


def main(args):
    model, dataset = args.model, args.dataset
    model_name = ClipModel.get_model_name_by_index(model)
    args.log_file = os.getcwd() + '/log/{}_{}_{}.txt'.format(args.mode, model_name, args.dataset)
    logger = get_logger(args.log_file, args.log_file)

    clip = ClipModel(model, args=args, logger=logger)  # 加载系统的clip模型时就是eval模式
    logger.info(f'Clip model {model_name} loaded')

    if args.mode == 'zs':  # zeroshot
        root = '../../data/'
        # 利用原有的数据加载器
        test_data = ImageTextData(args=args, dataset=args.test_data, root=root, preprocess=clip.preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False,
                                                  drop_last=False)
        acc, res = clip.evaluate(test_loader)
        logger.info('Results: {}'.format(res))
        logger.info('Accuracy: {:.2f}%'.format(acc * 100))
    elif args.mode == 'fe':  # feature extraction
        test_data = ImageTextData(args=args, dataset=args.test_data, root=args.root, preprocess=clip.preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False,
                                                  drop_last=False)
        res = clip.feature_extraction(test_loader)
        logger.info('Feature extracted!')
        if not os.path.exists('feat'):
            os.makedirs('feat')
        feat_file = 'feat/{}_{}_{}.csv'.format(args.mode, model_name, args.dataset)
        df = pd.DataFrame(res)
        df.to_csv(feat_file, index=True)  # index=False 参数指示不保存 DataFrame 的索引列
        # np.savetxt(feat_file, res, fmt='%.4f')

    elif args.mode == 'ft':  # fine-tuning 微调模式
        test_data = ImageTextData(args=args, dataset=args.test_data, root=args.root, preprocess=clip.preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False,
                                                  drop_last=False)
        optimizer = optim.AdamW(clip.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)
        best_acc = clip.finetune(trainloader=train_loader,
                                 testloader=test_loader,
                                 optimizer=optimizer,
                                 nepochs=args.nepoch,
                                 save_path='./log/{}_{}_{}.pt'.format(args.mode, model_name, args.dataset),
                                 schuse=args.schuse)

        logger.info('Accuracy: {:.2f}%'.format(best_acc * 100))

    elif args.mode == 'ftAdapter':
        wandb.init(
            project="Clip-Adapter",
            name="Clip{}-Cliplr{}-Basiclr{}-Ted{}".format(
                args.ClipMode,
                args.cliplr,
                args.basiclr,
                args.test_envs[0],
            ),
            config=vars(args)  # namespace to dict
        )
        # 准备数据集和对应的字典
        train_loaders, test_loaders = get_img_dataloader(args, clip_preprocess=clip.preprocess)
        eval_name_dict = train_valid_target_eval_names(args)

        # 准备模型
        params = list()
        if args.ClipMode == "Frozen":
            for name, paras in clip.model.named_parameters():  # 冻结clip模型的参数
                if 'logit_scale' in name:
                    paras.requires_grad = True
                else:
                    paras.requires_grad = False
        elif args.ClipMode == "Full":  # 训练整个clip模型
            for name, paras in clip.model.named_parameters():  # 冻结clip模型的参数
                paras.requires_grad = True
            params.append({"params": clip.model.parameters(), 'lr': args.cliplr})  # clip模型单独指定参数
        elif args.ClipMode == "Partial":
            clip_paras = list()
            for name, paras in clip.model.named_parameters():  # 只训练Clip模型中视觉层最后的attn模块
                if "visual.attnpool" in name:
                    # 设置param需要计算梯度
                    paras.requires_grad = True
                    clip_paras.append(paras)
                elif "transformer.resblocks.11" in name:  # 文本transformer的最后一个resblock
                    paras.requires_grad = True
                    clip_paras.append(paras)
                elif "logit_scale" in name and args.ContrastLossUse is True:
                    paras.requires_grad = True
                    clip_paras.append(paras)
                else:
                    paras.requires_grad = False
            params.append({'params': clip_paras})

        params.append({"params": clip.model.parameters(), 'lr': args.cliplr})  # clip模型单独指定参数
        params.append({"params": clip.ImgAdapter.parameters()})
        params.append({"params": clip.TextAdapter.parameters()})
        params.append({"params": clip.head.parameters()})

        optimizer = optim.AdamW(params=params, lr=args.basiclr, betas=(args.beta1, args.beta2),
                                eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)
        best_acc = clip.finetuneAdapter(trainloaders=train_loaders,
                                        testloaders=test_loaders,
                                        optimizer=optimizer,
                                        nepochs=args.nepoch,
                                        steps_per_epoch=args.steps_per_epoch,
                                        save_path='./log/{}_{}_{}.pt'.format(args.mode, model_name, args.dataset),
                                        eval_name_dict=eval_name_dict,
                                        schuse=args.schuse,
                                        mixupuse=args.mixupuse, )

        logger.info('Accuracy: {:.2f}%'.format(best_acc * 100))
    elif args.mode == 'fe_lp':  # fine-tuning 微调模式
        test_data = ImageTextData(args.test_data, root=args.root, preprocess=clip.preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False,
                                                  drop_last=False)
        # 优化图像适配器参数
        optimizer = optim.Adam(clip.clf.parameters(),
                               lr=args.lr, betas=(args.beta1, args.beta2),
                               eps=args.eps, weight_decay=args.weight_decay)
        # optimizer = optim.SGD(clip.clf.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        best_acc = clip.TrainLinearProbe(trainloader=train_loader,
                                         testloader=test_loader,
                                         optimizer=optimizer,
                                         nepochs=args.nepoch,
                                         save_path='./log/{}_{}_{}.pt'.format(args.mode, model_name, args.dataset),
                                         schuse=args.schuse)
        logger.info('Accuracy: {:.2f}%'.format(best_acc * 100))

    elif args.mode == 'lp':  # 提取特征后调用逻辑回归
        # -------------------------------------------------------------------------
        train_loaders, test_loaders = get_img_dataloader(args, clip_preprocess=clip.preprocess)
        eval_name_dict = train_valid_target_eval_names(args)
        print("Data Process Done!")
        acc_type_list = ['train', 'valid', 'target']
        train_feats, train_labels = None, None
        target_feats, target_labels = None, None
        for item in acc_type_list:  # 输出acc的过程计算较慢
            # eval_loaders[i]的索引i与eval_name_dict[item]的索引对应
            for i in eval_name_dict[item]:  # ['train', 'valid', 'target']
                # 计算对应domain下的准确率，然后取均值
                domain_dataloader = test_loaders[i]
                for batch in tqdm(domain_dataloader):
                    image, text, gt = batch
                    image = image.to(clip.device)
                    gt = gt.to(clip.device)

                    if item == "target":
                        if target_labels is None:
                            target_labels = gt
                        else:
                            target_labels = torch.cat([target_labels, gt], dim=0)
                    else:
                        if train_labels is None:
                            train_labels = torch.cat([gt, gt], dim=0)
                        else:
                            train_labels = torch.cat([train_labels, gt, gt], dim=0)

                    text = text.to(clip.device)
                    with torch.no_grad():
                        image_features = clip.model.encode_image(image)
                        text_features = clip.model.encode_text(text)

                        # normalized features
                        image_features = image_features / image_features.norm(dim=1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=1, keepdim=True)

                        if item == "target":
                            if target_feats is None:
                                target_feats = image_features
                            else:
                                target_feats = torch.cat([target_feats, image_features], dim=0)
                        else:
                            if train_feats is None:
                                train_feats = torch.cat([image_features, text_features], dim=0)
                            else:
                                train_feats = torch.cat([train_feats, image_features, text_features], dim=0)

        train_features = train_feats.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        test_features = target_feats.cpu().numpy()
        test_labels = target_labels.cpu().numpy()

        print("Feature Process Done!")
        # Perform logistic regression
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)

        # Evaluate using the logistic regression classifier
        train_predictions = classifier.predict(train_features)
        train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
        test_predictions = classifier.predict(test_features)
        test_accuracy = np.mean((test_labels == test_predictions).astype(float)) * 100.
        print(f"Train Accuracy = {train_accuracy:.3f}")  # A-->C调用逻辑回归也能达到Accuracy = 77.915%，
        print(f"Test Accuracy = {test_accuracy:.3f}")

    else:
        raise NotImplementedError


# def sweep_index(model=-1, data=-1):
#     if model == -1 and data == -1:
#         m_sweep_index = range(len(ClipModel.CLIP_MODELS))
#         d_sweep_index = range(len(ImageTextData._DATA_FOLDER))
#     elif model == -1 and data != -1:
#         m_sweep_index = range(len(ClipModel.CLIP_MODELS))
#         d_sweep_index = range(data, data + 1)
#     elif data == -1 and model != -1:
#         m_sweep_index = range(model, model + 1)
#         d_sweep_index = range(len(ImageTextData._DATA_FOLDER))
#     else:
#         m_sweep_index = range(model, model + 1)
#         d_sweep_index = range(data, data + 1)
#     return m_sweep_index, d_sweep_index


# def sweep(model=-1, data=-1):
#     m_sweep_index, d_sweep_index = sweep_index(model, data)
#     if args.result:
#         model_name_lst = [ClipModel.get_model_name_by_index(i) for i in m_sweep_index]
#         data_name_lst = [ImageTextData.get_data_name_by_index(i) for i in d_sweep_index]
#         res = gather_res(model_name_lst, data_name_lst)
#         for line in res:
#             print(line)
#     else:
#         for model in m_sweep_index:
#             for data in d_sweep_index:
#                 args.model = model
#                 args.dataset = data
#                 main(args)


# 更多关于Clip模型的应用解答
# https://github.com/openai/CLIP/issues/83

if __name__ == '__main__':
    args = get_args()
    set_gpu(args.gpu)
    set_seed(args.seed)
    main(args)
    # sweep(args.model, args.dataset)
