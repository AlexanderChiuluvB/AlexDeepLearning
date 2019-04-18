import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class ImageData(data.Dataset):

    def __init__(self,rootDir,transforms=None,train=True,test=False):

        self.test = test
        imgList = [os.path.join(rootDir,img) for img in os.listdir(rootDir)]


        if self.test:
            imgList = sorted(imgList, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgList = sorted(imgList, key=lambda x: int(x.split('.')[-2]))


        #划分数据集 train:val = 7:3

        imgNum = len(imgList)

        if self.test:
            self.imgs = imgList
        elif train:
            self.imgs = imgList[:imgNum]
        else:
            self.imgs = imgList[imgNum:]

        if transforms is None:

            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])
            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            # 训练集
            else :
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):


        """
        避免把所有图片都加到内存中,而是利用多进程

        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        """
        img_path = self.imgs[index]
        if self.test:
             label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
             label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        """
        返回数据集中所有图片个数
        """
        return len(self.imgs)
