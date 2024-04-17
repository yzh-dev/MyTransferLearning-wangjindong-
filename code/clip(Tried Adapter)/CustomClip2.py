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
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )
        # self.TransformerEncoder = nn.Transformer(d_model=c_in, nhead=8, num_encoder_layers=2)

    def forward(self, x):
        # a = self.fc(x)
        a = self.TransformerEncoder(x, x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
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
        self.ImgAdapter = Adapter(c_in=1024, reduction=4).to(self.model.dtype)
        self.ImgAdapter.train()
        self.ImgAdapter.to(device)
        self.TextAdapter = Adapter(c_in=1024, reduction=4).to(self.model.dtype)
        self.TextAdapter.train()
        self.TextAdapter.to(device)
        self.head = nn.Linear(1024, 65).to(self.model.dtype)
        self.head.train()
        self.head.to(device)
        self.discriminator = Adver_network.Discriminator(input_dim=1024,
                                                         hidden_dim=256,
                                                         num_domains=len(args.domains) - len(args.test_envs))
        self.discriminator.train()
        self.discriminator.to(device)
        # 初始化为0.1，学习率可以设置的更高一些，因为这个参数是用来平衡两个loss的
        self.lam = torch.nn.Parameter(torch.tensor([0.5], device=device))  # 定义一个可学习参数, 用于平衡两个loss
        # 实验结果表明，gamma的值如果是学习，会稳定为负值，难以解释，所以不再采用对抗域的方式
        self.mmd_gamma= torch.nn.Parameter(torch.tensor([0.1], device=device))  # 定义一个可学习参数, 用于平衡两个loss

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

    # 输入多模态的imgs和texts，获取到编码后的feats
    def getCrossModalFeat(self, imgs, texts):
        image_features = self.model.encode_image(imgs)
        text_features = self.model.encode_text(texts)
        # 测试是否使用了适配器
        if self.args.ImgAdapUse:
            image_features = self.ImgAdapter(image_features)
        if self.args.TxtAdapUse:
            text_features = self.TextAdapter(text_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return image_features, text_features

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

    # CoralLoss
    def coral(self, d1_feat, d2_feat):
        mean_d1_feat = d1_feat.mean(0, keepdim=True)
        mean_d2_feat = d2_feat.mean(0, keepdim=True)
        cent_d1_feat = d1_feat - mean_d1_feat
        cent_d2_feat = d2_feat - mean_d2_feat
        cova_d1_feat = (cent_d1_feat.t() @ cent_d1_feat) / (len(d1_feat) - 1)
        cova_d2_feat = (cent_d2_feat.t() @ cent_d2_feat) / (len(d2_feat) - 1)

        mean_diff = (mean_d1_feat - mean_d2_feat).pow(2).mean()
        cova_diff = (cova_d1_feat - cova_d2_feat).pow(2).mean()

        return mean_diff + cova_diff

    # 微调适配器，跨模态模式
    def finetuneAdapter(self, trainloaders, testloaders, optimizer, nepochs=10, steps_per_epoch=100, save_path=None,
                        eval_name_dict=None, schuse=None, mixupuse=None):
        best_acc = 0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=nepochs * steps_per_epoch, )

        if self.args.WarmUpUse:
            warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer=optimizer)  # 如果不使用warmup，可以注释掉
        train_minibatches_iterator = zip(*trainloaders)
        # initialize the early_stopping object
        # early_stopping = EarlyStopping(patience=5, verbose=True)#如果5个epoch没有提升，就停止训练
        for epoch in range(nepochs):
            total_loss = 0
            list1 = list(range(steps_per_epoch))
            for iter_num in tqdm(list1):
                optimizer.zero_grad()
                minibatches_device = [(data) for data in next(train_minibatches_iterator)]  # 获取不同domain的数据

                # 如果是使用mixup
                if mixupuse is True:
                    # 计算分类损失
                    cls_loss = 0
                    disc_image_features = None
                    for (imgi, txti, labeli), (imgj, txtj, labelj) in random_pairs_of_minibatches(minibatches_device):
                        # 获得一个domain下的数据(imgi, txti, labeli)，其中(imgj, txtj, labelj)是另一个domain下的数据，将两者混合
                        mixupalpha = 0.2
                        # imgi： batch*3*224*224
                        # txti： batch*77
                        # labeli： batch*1

                        imgi = imgi.to(self.device)
                        txti = txti.to(self.device)
                        labeli = labeli.to(self.device)

                        imgi_features = self.model.encode_image(imgi)
                        imgi_features = self.ImgAdapter(imgi_features)
                        txti_features = self.model.encode_text(txti)
                        txti_features = self.TextAdapter(txti_features)

                        imgi_features = imgi_features / imgi_features.norm(dim=1, keepdim=True)
                        txti_features = txti_features / txti_features.norm(dim=1, keepdim=True)
                        cross_model_featsi = torch.cat((imgi_features, txti_features), 0)

                        if disc_image_features is None:
                            disc_image_features = imgi_features
                        else:
                            disc_image_features = torch.cat((disc_image_features, imgi_features), 0)

                        imgj = imgj.to(self.device)
                        txtj = txtj.to(self.device)
                        labelj = labelj.to(self.device)

                        imgj_features = self.model.encode_image(imgj)
                        imgj_features = self.ImgAdapter(imgj_features)
                        txtj_features = self.model.encode_text(txtj)
                        txtj_features = self.TextAdapter(txtj_features)

                        imgj_features = imgj_features / imgj_features.norm(dim=1, keepdim=True)
                        txtj_features = txtj_features / txtj_features.norm(dim=1, keepdim=True)
                        cross_model_featsj = torch.cat((imgj_features, txtj_features), 0)

                        lam = np.random.beta(mixupalpha, mixupalpha)
                        # mixup变换后的领域不变特征
                        cross_model_feats = (lam * cross_model_featsi + (1 - lam) * cross_model_featsj)

                        logits = self.head(cross_model_feats)  # batch*65
                        logits = self.model.logit_scale.exp() * logits

                        labeli_gt = torch.cat([labeli, labeli], dim=0)
                        labelj_gt = torch.cat([labelj, labelj], dim=0)

                        cls_loss += lam * torch.nn.functional.cross_entropy(logits, labeli_gt.cuda().long())
                        cls_loss += (1 - lam) * torch.nn.functional.cross_entropy(logits, labelj_gt.cuda().long())

                    # 计算领域判别器损失，领域判别器的输入image模态的数据
                    disc_input = Adver_network.ReverseLayerF.apply(disc_image_features, self.alpha)
                    clip.model.convert_weights(self.discriminator)  # 将模型参数转换为fp16
                    disc_out = self.discriminator(disc_input)
                    disc_labels = None  # 领域标签gt
                    for i in range(len(minibatches_device)):
                        gt = torch.zeros(len(minibatches_device[i][0])).fill_(i)
                        if disc_labels is None:
                            disc_labels = gt
                        else:
                            disc_labels = torch.cat((disc_labels, gt), dim=0)
                    disc_labels = disc_labels.to(self.device).long()
                    disc_loss = torch.nn.functional.cross_entropy(disc_out, disc_labels)
                    loss = cls_loss + disc_loss
                    loss /= len(minibatches_device)

                    loss.backward()
                    total_loss += loss.item()

                # 不使用mixup，直接提取各个域的数据
                else:
                    all_images = torch.cat([data[0].cuda().float() for data in minibatches_device])
                    all_texts = torch.cat([data[1].cuda().long() for data in minibatches_device])
                    all_gts = torch.cat([data[2].cuda().long() for data in minibatches_device])

                    all_images = all_images.to(self.device)
                    all_texts = all_texts.to(self.device)
                    all_gts = all_gts.to(self.device)
                    cross_model_gt = torch.cat([all_gts, all_gts], dim=0)
                    image_features = self.model.encode_image(all_images)
                    text_features = self.model.encode_text(all_texts)
                    # 测试是否使用了适配器
                    if self.args.ImgAdapUse:
                        image_features = self.ImgAdapter(image_features)
                    if self.args.TxtAdapUse:
                        text_features = self.TextAdapter(text_features)

                    # normalized features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    cross_model_feats = torch.cat([image_features, text_features], dim=0)

                    if self.args.ContrastLossUse:#采用对比损失
                        # cosine similarity as logits
                        logit_scale = self.model.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()
                        ground_truth = torch.arange(len(all_images), dtype=torch.long, device=self.device)
                        cls_loss = (torch.nn.functional.cross_entropy(logits_per_image, ground_truth) + torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2
                    else:#采用交叉熵损失
                        logits = self.head(cross_model_feats)  # batch*65
                        cls_loss = torch.nn.functional.cross_entropy(logits, cross_model_gt)

                    # 如果进行领域判别器训练
                    if self.args.dduse:
                        # 计算领域判别器损失，领域判别器的输入image模态的数据
                        disc_input = Adver_network.ReverseLayerF.apply(image_features, self.alpha)
                        clip.model.convert_weights(self.discriminator)  # 将模型参数转换为fp16
                        disc_out = self.discriminator(disc_input)
                        disc_labels = None  # 领域标签gt
                        for i in range(len(minibatches_device)):
                            gt = torch.zeros(len(minibatches_device[i][0])).fill_(i)
                            if disc_labels is None:
                                disc_labels = gt
                            else:
                                disc_labels = torch.cat((disc_labels, gt), dim=0)
                        disc_labels = disc_labels.to(self.device).long()
                        disc_loss = torch.nn.functional.cross_entropy(disc_out, disc_labels)
                        loss = cls_loss + disc_loss.detach().item()

                    else:
                        loss = cls_loss

                    nmm_domains = len(minibatches_device)
                    penalty = 0
                    batch = self.args.batchsize
                    image_feats = [image_features[i * batch:(i + 1) * batch, :] for i in range(4)]
                    # 计算coral损失
                    if self.args.coraluse:
                        for i in range(nmm_domains):
                            for j in range(i + 1, nmm_domains):
                                penalty += self.coral(image_feats[i], image_feats[j])
                        if nmm_domains > 1:
                            penalty /= (nmm_domains * (nmm_domains - 1) / 2)
                        loss = loss+self.mmd_gamma*penalty

                    loss.backward()
                    total_loss += loss.detach().item()

                if self.args.ClipMode == "Frozen":
                    # Clip模型不需训练
                    optimizer.step()
                else:  # 完全微调Clip
                    convert_models_to_fp32(args=self.args, model=self.model)
                    optimizer.step()
                    # clip.model.convert_weights basically convert the CLIP model weight into float16. This will help accelerate and reduce memory usage during training.
                    # The definition of clip.model.convert_weight can be found at https://github.com/openai/CLIP/blob/main/clip/model.py line 371
                    clip.model.convert_weights(self.model)

                # lr_scheduler.step()
                if self.args.WarmUpUse:
                    with warmup_scheduler.dampening():
                        lr_scheduler.step()
                else:
                    lr_scheduler.step()

            acc_record, valid_loss = self.evaluateAdapter(testloaders, eval_name_dict=eval_name_dict)
            if acc_record["valid"] > best_acc:
                best_acc = acc_record["valid"]
                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)
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
                # 'cliplr': optimizer.param_groups[0]['lr'],#clip的学习率
                'headlr': optimizer.param_groups[-1]['lr'],  # 预测头head的学习率
                # 'mmd_gamma': self.mmd_gamma.item(),  # 预测头head的学习率
            })
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            # early_stopping(valid_loss, self.model)

            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
        return best_acc

    def evaluateAdapter(self, dataloaders, eval_name_dict=None, modelpath=None):
        if modelpath is not None:
            self.model.load_state_dict(torch.load(modelpath))
        # 设置为评估模式，关闭dropout
        self.ImgAdapter.eval()

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
                    image, _, gt = batch
                    image = image.to(self.device)
                    gt = gt.to(self.device)

                    if self.args.ContrastLossUse:  # 采用对比损失
                        texts = domain_dataloader.dataset.str_labels
                        text_features = self.get_text_features_list(texts)
                        with torch.no_grad():
                            image_features = self.model.encode_image(image)
                            if self.args.ImgAdapUse:
                                image_features = self.ImgAdapter(image_features)
                            # normalized features
                            image_features = image_features / image_features.norm(dim=1, keepdim=True)
                            similarity = self.get_similarity(image_features, text_features)
                            _, indices = similarity.topk(1)

                            pred = torch.squeeze(indices)
                            result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
                    else:
                        with torch.no_grad():
                            # -----------------------------------------------------------
                            # 统计准确率时，只能计算图片模态的准确率
                            image_features = self.model.encode_image(image)
                            if self.args.ImgAdapUse:
                                image_features = self.ImgAdapter(image_features)
                            # normalized features
                            image_features = image_features / image_features.norm(dim=1, keepdim=True)

                            logits = self.head(image_features)  # batch*65
                            preds = logits.argmax(dim=1)
                            result = torch.cat([preds.view(-1, 1), gt.view(-1, 1)], dim=1)
                            if item == "valid":
                                cls_loss = torch.nn.functional.cross_entropy(logits, gt)
                                cls_loss /= len(domain_dataloader)
                                valid_loss += cls_loss.detach().item()

                    if res is None:
                        res = result
                    else:
                        res = torch.cat([res, result], dim=0)
            res = res.cpu().numpy()
            acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
            acc_record[item] = acc
        valid_loss /= len(eval_name_dict["valid"])
        self.ImgAdapter.train()
        return acc_record, valid_loss
