import logging

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

logger = logging.getLogger(__name__)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


class MNISTSSL(datasets.MNIST):
    def __init__(self, root, unlabeled_indexs=None, labeled_indexs=None, labeled_targets=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.train=train
        self.unlabeled_indexs = unlabeled_indexs

        if unlabeled_indexs is not None:
            self.data = self.data[unlabeled_indexs]
            self.data_id = unlabeled_indexs
            self.targets = np.array(self.targets)[unlabeled_indexs]

        if (labeled_indexs is not None) and (labeled_targets is not None):
            real = np.array(self.targets)[labeled_indexs]
            self.data = self.data[labeled_indexs]
            self.targets = np.array(labeled_targets)
            print(f'acc:  {(self.targets == real).sum() / len(self.targets) * 100}%')


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = Image.fromarray(img.numpy())
        # if img.mode == 'L':
        #     img = img.convert('RGB')

        img = img.numpy()
        img= np.expand_dims(img,2).repeat(3, axis=2)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.unlabeled_indexs is not None:
            return img, target,self.data_id[index]
        else:
            return img, target


