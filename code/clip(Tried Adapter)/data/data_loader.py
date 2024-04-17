import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import os
import clip
import numpy as np
from PIL import ImageFile, Image
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DGImageTextData(object):

    def __init__(self, dataset, rootdir, domain_name, preprocess, indices=None, test_envs=[], transform=None,
                 prompt='a picture of a', promptPooling=None):
        self.dataset = dataset
        path = os.path.join(rootdir, dataset, domain_name)
        data = ImageFolder(root=path, transform=transform)
        self.data = data

        int_labels = [item[1] for item in self.data.imgs]
        self.int_labels = np.array(int_labels)
        str_labels = data.classes
        self.str_labels = str_labels

        if indices is None:
            self.indices = np.arange(len(self.data.imgs))
        else:
            self.indices = indices

        self.preprocess = preprocess
        self.promptPooling = promptPooling
        # prompt = 'a {} of a {}'
        # self.str_labels = [prompt.format(domain_name,x) for x in self.str_labels]  # dataset添加了prompt的标签
        self.str_labels = [prompt + ' ' + x for x in self.str_labels]  # dataset添加了prompt的标签
        self.text_encs = clip.tokenize(self.str_labels)

    def __getitem__(self, index):
        index = self.indices[index]
        image, int_label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))

        text_enc = list()
        if self.promptPooling:
            for enc in self.text_encs:
                text_enc.append(enc[int_label])  # 利用每一种prompt编码方式进行编码
        else:
            text_enc = self.text_encs[int_label]  # 获取文本编码
        return image, text_enc, int_label

    def __len__(self):
        return len(self.indices)

    # 默认的数据转换方式
    # _TRANSFORM = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])


class ImageTextData(object):

    def __init__(self, args, dataset, root, preprocess, prompt='a picture of a'):
        if type(dataset) is int:
            dataset = self._DATA_FOLDER[dataset]
        dataset = os.path.join(root, dataset)
        if dataset == 'imagenet-r':
            data = datasets.ImageFolder(
                'imagenet-r', transform=self._TRANSFORM)
            labels = open('imagenetr_labels.txt').read().splitlines()
            labels = [x.split(',')[1].strip() for x in labels]
        else:
            data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
            labels = data.classes
        self.data = data
        self.labels = labels
        if prompt:
            self.labels = [prompt + ' ' + x for x in self.labels]  # dataset添加了prompt的标签

        self.preprocess = preprocess
        # 通过clip获取文本编码，在这里进行prompt学习
        self.text = clip.tokenize(self.labels)  # Returns the tokenized representation of given input string(s)

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        text_enc = self.text[label]  # 获取文本编码
        return image, text_enc, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _DATA_FOLDER = [
        'dataset/OfficeHome/Art',
        'dataset/OfficeHome/Clipart',
        'dataset/OfficeHome/Product',
        'dataset/OfficeHome/RealWorld',

        'dataset/office31/amazon',  # 4
        'dataset/office31/webcam',
        'dataset/office31/dslr',

        'dataset/VLCS/Caltech101',  # 7
        'dataset/VLCS/LabelMe',
        'dataset/VLCS/SUN09',
        'dataset/VLCS/VOC2007',

        'dataset/PACS/kfold/art_painting',  # 11
        'dataset/PACS/kfold/cartoon',
        'dataset/PACS/kfold/photo',
        'dataset/PACS/kfold/sketch',

        'dataset/visda/validation',  # 15

        'dataset/domainnet/clipart',  # 16
        'dataset/domainnet/infograph',
        'dataset/domainnet/painting',
        'dataset/domainnet/quickdraw',
        'dataset/domainnet/real',
        'dataset/domainnet/sketch',

        'dataset/terra_incognita/location_38',  # 22
        'dataset/terra_incognita/location_43',
        'dataset/terra_incognita/location_46',
        'dataset/terra_incognita/location_100',

        'imagenet-r',
    ]
    # 默认的数据转换方式
    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class ImageTextFeatData(object):
    def __init__(self, path):
        # 读取path路径中的csv数据，不要第一行和第一列
        train_data = pd.read_csv(path, header=None)
        train_res = pd.DataFrame(train_data, index=None)
        # 获取特征和标签
        image_feats = train_res.iloc[:, :-1].values
        labels = train_res.iloc[:, -1].values
        self.image_feats = torch.tensor(image_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.image_feats[index], self.labels[index]

    def __len__(self):
        return len(self.image_feats)

# if __name__ == '__main__':
#     print(ImageTextData.get_data_name_by_index(0))
