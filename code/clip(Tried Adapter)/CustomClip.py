import clip
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data.getdataloader import random_pairs_of_minibatches
from network import Adver_network
from utils import convert_models_to_fp32
import wandb
import pytorch_warmup as warmup
from pytorchtools import EarlyStopping


# Clip适配器
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(c_in // reduction, c_in, bias=False),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        a = self.fc(x)
        # x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        x = a + x
        return x


class ClipModel(object):
    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', args=None, device='cuda', logger=None):
        self.args = args
        self.alpha = args.alpha
        self.device = device
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        # 添加适配器
        self.ImgAdapter = Adapter(c_in=1024, reduction=8).to(self.model.dtype)
        self.ImgAdapter.train()
        self.ImgAdapter.to(device)
        self.TextAdapter = Adapter(c_in=1024, reduction=8).to(self.model.dtype)
        self.TextAdapter.train()
        self.TextAdapter.to(device)

        self.head = nn.Linear(1024, 65).to(self.model.dtype)
        self.head.train()
        self.head.to(device)

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModel.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def get_image_features(self, image, need_preprocess=False):
        if need_preprocess:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def get_text_feature(self, text):
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features

    def get_text_features_list(self, texts, train=False):
        if train:
            text_inputs = torch.cat([clip.tokenize(c)
                                     for c in texts]).to(self.device)
            text_features = self.model.encode_text(text_inputs)
        else:
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(c)
                                         for c in texts]).to(self.device)
                text_features = self.model.encode_text(text_inputs)

        return text_features

    def get_similarity(self, image_features, text_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity

    def get_topk(self, image, text, k=1):
        similarity = self.get_similarity(image, text)
        values, indices = similarity[0].topk(k)
        return values, indices

    def feature_extraction(self, dataloader):
        res = None
        for batch in tqdm(dataloader):
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = self.get_image_features(image)
            feat_lab = torch.cat(
                [image_features, label.view(-1, 1)], dim=1)
            if res is None:
                res = torch.zeros((1, feat_lab.shape[1])).to(self.device)
            res = torch.cat([res, feat_lab], dim=0)
        res = res[1:, :].cpu().numpy()
        return res

    def finetune(self, trainloader, testloader, optimizer, nepochs=10, save_path=None, schuse=None):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        best_acc = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=nepochs * len(trainloader))
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        for epoch in range(nepochs):
            train_res = None
            total_loss = 0
            all_labels = trainloader.dataset.labels
            all_label_features = self.get_text_features_list(all_labels)  # 所有标签的特征

            for batch in tqdm(trainloader):
                optimizer.zero_grad()
                image, text, gt = batch
                image = image.to(self.device)
                text = text.to(self.device)
                gt = gt.to(self.device)

                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                # ------------------------------------------------------------
                # 统计训练集上的分类准确率
                image_features2 = image_features.clone()
                all_label_features /= all_label_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features2 @ all_label_features.T).softmax(dim=-1)
                _, indices = similarity.topk(1)

                pred = torch.squeeze(indices)
                result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
                if train_res is None:
                    train_res = result
                else:
                    train_res = torch.cat([train_res, result], dim=0)

                # -----------------------------------------------------------
                # 统计训练集上的loss
                ground_truth = torch.arange(len(image), dtype=torch.long, device=self.device)
                loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

                loss.backward()
                total_loss += loss.item()
                if self.device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(self.model)
                    optimizer.step()
                    clip.model.convert_weights(self.model)

                if schuse:
                    scheduler.step()
                    # with warmup_scheduler.dampening():
                    #     scheduler.step()

            train_res = train_res.cpu().numpy()
            train_acc = np.mean(np.array(train_res)[:, 0] == np.array(train_res)[:, 1])
            eval_acc, _ = self.evaluate(testloader)
            if eval_acc > best_acc:
                best_acc = eval_acc
                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)
            self.logger.info("Epoch {} : TrainLoss {}, TrainAcc {:.4f}, EvalAcc {:.4f}".
                             format(epoch, total_loss / len(trainloader), train_acc, eval_acc))
            wandb.log({'TrainLoss': total_loss / len(trainloader),
                       'TrainAcc': train_acc,
                       'EvalAcc': eval_acc})
        return best_acc

    def evaluate(self, dataloader, modelpath=None):
        if modelpath is not None:
            self.model.load_state_dict(torch.load(modelpath))

        texts = dataloader.dataset.labels
        text_features = self.get_text_features_list(texts)
        res = None

        for batch in tqdm(dataloader):
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = self.get_image_features(image)
            similarity = self.get_similarity(image_features, text_features)
            _, indices = similarity.topk(1)

            pred = torch.squeeze(indices)
            result = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            if res is None:
                res = result
            else:
                res = torch.cat([res, result], dim=0)
        res = res.cpu().numpy()
        acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
        return acc, res

    # 微调适配器，跨模态模式
    def finetuneAdapter(self, trainloaders, testloaders, optimizer, nepochs=10, steps_per_epoch=100, save_path=None,
                        eval_name_dict=None, schuse=None, mixupuse=None):
        best_acc = 0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=nepochs * steps_per_epoch, )
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer=optimizer)
        train_minibatches_iterator = zip(*trainloaders)
        for epoch in range(nepochs):
            total_loss = 0
            list1 = list(range(steps_per_epoch))
            for iter_num in tqdm(list1):
                optimizer.zero_grad()
                minibatches_device = [(data) for data in next(train_minibatches_iterator)]  # 获取不同domain的数据

                all_images = torch.cat([data[0].cuda().float() for data in minibatches_device])
                all_texts = torch.cat([data[1].cuda().long() for data in minibatches_device])
                # all_gts = torch.cat([data[2].cuda().long() for data in minibatches_device])

                all_images = all_images.to(self.device)
                all_texts = all_texts.to(self.device)
                # all_gts = all_gts.to(self.device)
                # cross_model_gt = torch.cat([all_gts, all_gts], dim=0)
                image_features = self.model.encode_image(all_images)
                image_features = self.ImgAdapter(image_features)
                text_features = self.model.encode_text(all_texts)
                text_features = self.TextAdapter(text_features)

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                ground_truth = torch.arange(len(all_images), dtype=torch.long, device=self.device)
                cls_loss = (torch.nn.functional.cross_entropy(logits_per_image,ground_truth) +
                            torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2

                loss = cls_loss
                loss.backward()
                total_loss += loss.item()

                convert_models_to_fp32(args=self.args, model=self.model)
                optimizer.step()
                clip.model.convert_weights(self.model)

                with warmup_scheduler.dampening():
                    lr_scheduler.step()

            acc_record, valid_loss = self.evaluateAdapter(testloaders, eval_name_dict=eval_name_dict)
            if acc_record["valid"] > best_acc:
                best_acc = acc_record["valid"]
            self.logger.info("Epoch {} : TrainLoss {}, ValLoss {},TrainAcc {:.4f}, ValAcc {:.4f},TargetAcc {:.4f}".
                             format(epoch,
                                    total_loss / steps_per_epoch,
                                    valid_loss,
                                    acc_record["train"],
                                    acc_record["valid"],
                                    acc_record["target"]))
            wandb.log({
                'TrainLoss': total_loss / steps_per_epoch,
                'ValLoss': valid_loss,
                'TrainAcc': acc_record["train"],
                'ValAcc': acc_record["valid"],
                'TargetAcc': acc_record["target"],
                'headlr': optimizer.param_groups[-1]['lr'],  # 预测头head的学习率
            })
        return best_acc

    def evaluateAdapter(self, dataloaders, eval_name_dict=None, modelpath=None):
        self.ImgAdapter.eval()
        self.TextAdapter.eval()
        acc_record = {}
        valid_loss = 0
        acc_type_list = ['train', 'valid', 'target']
        for item in acc_type_list:  # ['train', 'valid', 'target']
            # eval_loaders[i]的索引i与eval_name_dict[item]的索引对应
            res = None
            for i in eval_name_dict[item]:
                # 计算对应domain下的准确率，然后取均值
                domain_dataloader = dataloaders[i]
                for batch in tqdm(domain_dataloader):
                    image, _, gt_label = batch
                    image = image.to(self.device)
                    gt_label = gt_label.to(self.device)

                    with torch.no_grad():
                        image_features = self.model.encode_image(image)
                        image_features = self.ImgAdapter(image_features)
                        texts = domain_dataloader.dataset.str_labels
                        text_features = self.get_text_features_list(texts)
                        text_features = self.TextAdapter(text_features)

                        # normalized features
                        similarity = self.get_similarity(image_features, text_features)
                        _, indices = similarity.topk(1)
                        pred = torch.squeeze(indices)
                        result = torch.cat([pred.view(-1, 1), gt_label.view(-1, 1)], dim=1)

                    if res is None:
                        res = result
                    else:
                        res = torch.cat([res, result], dim=0)
            res = res.cpu().numpy()
            acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
            acc_record[item] = acc
        valid_loss /= len(eval_name_dict["valid"])
        self.ImgAdapter.train()
        self.TextAdapter.train()
        return acc_record, valid_loss
