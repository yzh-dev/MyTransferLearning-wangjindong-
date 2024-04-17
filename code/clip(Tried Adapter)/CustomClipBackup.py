import clip
import wandb
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_warmup as warmup
from utils import convert_models_to_fp32


# 适配器
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# 线性分类头
class LinearProbe(nn.Module):
    def __init__(self, c_in, c_out, reduction=4,device='cuda', logger=None):
        super(LinearProbe, self).__init__()
        self.device = device
        self.logger = logger
        self.MLP = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.BatchNorm1d(c_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_out),
        )

        for m in self.MLP.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.MLP(x)
        return x

    def TrainLinearProbe(self, trainloader, testloader, optimizer, nepochs=10, save_path=None, schuse=None):
        clf_loss = nn.CrossEntropyLoss()
        best_acc = 0


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=nepochs)
        for epoch in range(nepochs):
            self.MLP.train()  # 将分类器设置为训练模式
            self.MLP = self.MLP.to(self.device)
            train_res = None
            total_loss = 0
            for batch in tqdm(trainloader):
                optimizer.zero_grad()
                image_feat,  gt = batch
                image_feat = image_feat.to(self.device)
                gt = gt.to(self.device)
                image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
                logits = self.MLP(image_feat)

                _, pred = torch.max(logits, dim=1)
                result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
                if train_res is None:
                    train_res = result
                else:
                    train_res = torch.cat([train_res, result], dim=0)

                loss = clf_loss(logits, gt)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            # 每个epoch结束后，调整下学习率
            if schuse:
                scheduler.step()

            train_res = train_res.cpu().numpy()
            train_acc = np.mean(np.array(train_res)[:, 0] == np.array(train_res)[:, 1])
            eval_acc, _ = self.EvaluateLinearProbe(testloader)
            if eval_acc > best_acc:
                best_acc = eval_acc
                if save_path is not None:
                    torch.save(self.MLP.state_dict(), save_path)
            self.logger.info("Epoch {} : TrainLoss {}, TrainAcc {:.4f}, EvalAcc {:.4f}".
                             format(epoch, total_loss / len(trainloader), train_acc, eval_acc))
            wandb.log({'TrainLoss': total_loss / len(trainloader),
                       'TrainAcc': train_acc,
                       'EvalAcc': eval_acc})

        return best_acc

    def EvaluateLinearProbe(self, dataloader, modelpath=None):
        if modelpath is not None:
            self.model.load_state_dict(torch.load(modelpath))

        self.MLP.eval()
        self.MLP = self.MLP.to(self.device)
        res = None
        for batch in tqdm(dataloader):
            image_feat, gt = batch
            image_feat = image_feat.to(self.device)
            gt = gt.to(self.device)
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            logits = self.MLP(image_feat)
            _, pred = torch.max(logits, dim=1)
            result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
            if res is None:
                res = result
            else:
                res = torch.cat([res, result], dim=0)
        res = res.cpu().numpy()
        acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
        return acc, res

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

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None):
        self.device = device
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        # 添加一个视觉适配器
        # self.ImgAdapter = Adapter(1024, 4).to(self.model.dtype)
        # self.ImgAdapter.to(device)
        # self.clf = LinearProbe(1024, 65).to(self.model.dtype)  # 输入为图像特征的维度，输出为类别数
        # self.clf.to(device)

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
            image_features = self.get_image_features(image)  # (batch_size, 1024)，只提取了图片的特征
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

        # 只训练Clip模型的视觉编码器
        for name, param in self.model.named_parameters():
            if 'visual' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=nepochs)
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        for epoch in range(nepochs):
            train_res = None
            total_loss = 0
            # 训练集上进行训练
            #model.train()的作用是启用 Batch Normalization 和 Dropout
            # Clip对超参数比较敏感，https://github.com/openai/CLIP/issues/150建议再训练的时候也设置为model.eval()
            self.model.eval()

            all_labels = trainloader.dataset.labels
            all_label_features = self.get_text_features_list(all_labels)  # 所有标签的特征

            for batch in tqdm(trainloader):
                optimizer.zero_grad()
                image, text, gt = batch
                image = image.to(self.device)
                text = text.to(self.device)
                gt = gt.to(self.device)

                # -----------------------------------------------------------
                # 原始步骤，直接通过视觉适配器和文本适配器得到特征
                # logits_per_image, logits_per_text = self.model(image, text)
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                #------------------------------------------------------------
                # 统计训练集上的分类准确率
                similarity = self.get_similarity(image_features, all_label_features)  # batch_size*num_labels
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

                optimizer.step()
                if self.device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(self.model)
                    optimizer.step()
                    # clip.model.convert_weights basically convert the CLIP model weight into float16. This will help accelerate and reduce memory usage during training.
                    # The definition of clip.model.convert_weight can be found at https://github.com/openai/CLIP/blob/main/clip/model.py line 371
                    clip.model.convert_weights(self.model)

            # 每个epoch结束后，调整下学习率
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

    # Clip模型容易出现灾难性遗忘，最好是保持其视觉编码器和文本编码器不变
    def TrainLinearProbe(self, trainloader, testloader, optimizer, nepochs=10, save_path=None,schuse=None):
        clf_loss = nn.CrossEntropyLoss()
        best_acc = 0

        # 冻结Clip模型的参数
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=nepochs)
        for epoch in range(nepochs):
            self.clf.train()  # 将分类器设置为训练模式
            train_res = None
            total_loss = 0
            for batch in tqdm(trainloader):
                optimizer.zero_grad()
                image, text, gt = batch
                image = image.to(self.device)
                text = text.to(self.device)
                gt = gt.to(self.device)
                image_features = self.model.encode_image(image)  # 提取图像特征
                # text_features = self.model.encode_text(text)  # 提取文本特征
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                # text_features = text_features / text_features.norm(dim=1, keepdim=True)
                # image_text_features = torch.cat([image_features, text_features], dim=1)  # 拼接图像、文本特征
                logits = self.clf(image_features)

                _, pred = torch.max(logits, dim=1)
                result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
                if train_res is None:
                    train_res = result
                else:
                    train_res = torch.cat([train_res, result], dim=0)

                loss = clf_loss(logits, gt)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            # 每个epoch结束后，调整下学习率
            if schuse:
                scheduler.step()
            # 手动调整学习率,在训练进度90%和70%的时候降低学习率
            # if (epoch in [int(nepochs * 0.3), int(nepochs * 0.5)]) and (not schuse):
            #     print('manually descrease lr')
            #     for params in optimizer.param_groups:
            #         params['lr'] = params['lr'] * 0.1

            train_res = train_res.cpu().numpy()
            train_acc = np.mean(np.array(train_res)[:, 0] == np.array(train_res)[:, 1])
            eval_acc, _ = self.EvaluateLinearProbe(testloader)
            if eval_acc > best_acc:
                best_acc = eval_acc
                if save_path is not None:
                    torch.save(self.clf.state_dict(), save_path)
            self.logger.info("Epoch {} : TrainLoss {}, TrainAcc {:.4f}, EvalAcc {:.4f}".
                             format(epoch, total_loss / len(trainloader), train_acc, eval_acc))
            wandb.log({'TrainLoss': total_loss / len(trainloader),
                       'TrainAcc': train_acc,
                       'EvalAcc': eval_acc})

        return best_acc

    def EvaluateLinearProbe(self, dataloader, modelpath=None):
        if modelpath is not None:
            self.model.load_state_dict(torch.load(modelpath))

        self.clf.eval()
        res = None
        for batch in tqdm(dataloader):
            image, text, gt = batch
            image = image.to(self.device)
            text = text.to(self.device)
            gt = gt.to(self.device)
            image_features = self.model.encode_image(image)
            # text_features = self.model.encode_text(text)
            # # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # # 拼接图像、文本特征
            # image_text_features = torch.cat([image_features, text_features], dim=1)
            logits = self.clf(image_features)
            _, pred = torch.max(logits, dim=1)
            result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
            if res is None:
                res = result
            else:
                res = torch.cat([res, result], dim=0)
        res = res.cpu().numpy()
        acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
        return acc, res

    def evaluate(self, dataloader, modelpath=None):
        if modelpath is not None:
            self.model.load_state_dict(torch.load(modelpath))
        labels = dataloader.dataset.labels
        label_features = self.get_text_features_list(labels)  # 获取到标签的特征,这里的标签特征是固定的,num_labels*1024
        res = None
        self.model.eval()
        for batch in tqdm(dataloader):
            image, _, gt = batch
            image = image.to(self.device)
            gt = gt.to(self.device)
            image_features = self.get_image_features(image)  # batch_size*1024

            similarity = self.get_similarity(image_features, label_features)  # batch_size*num_labels
            _, indices = similarity.topk(1)

            pred = torch.squeeze(indices)
            result = torch.cat([pred.view(-1, 1), gt.view(-1, 1)], dim=1)
            if res is None:
                res = result
            else:
                res = torch.cat([res, result], dim=0)
        res = res.cpu().numpy()
        acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
        return acc, res


if __name__ == '__main__':
    print(ClipModel.CLIP_MODELS)
    model_name = 'ViT-B/32'
    model_name = 5
    device = 'cuda'
    clip_inference = CLIP_INFERENCE(model_name, device)  # type: ignore
    print(clip_inference.model_name)

    image = Image.open('../test.jpg')
    text = 'a picture of a cat'
    print(clip_inference.inference(image, text))  # type: ignore

    dataset = datasets.ImageFolder(
        root='../dataset/office31/amazon',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    )
    labels = dataset.classes
    res, acc = clip_inference.classification(dataset, labels)
    print(res)
    print(acc)
