import sys
import os
import logging
import torchvision.transforms as transforms
import dataloader.transform as owntransform
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader.cifar import CIFAR10SSL, CIFAR100SSL
from dataloader.SVHN import SVHNSSL
from dataloader.STL10 import STL10SSL
from dataloader.MNIST import MNISTSSL
import torchvision
import numpy as np
from dataloader.randaugment import RandomAugment
from config import  cfg
from dataloader.setting import cfg_data
from .sampler import RandomSampler, BatchSampler
import copy
mean_std = cfg_data.MEAN_STD

val_transform =owntransform.Compose([
        owntransform.ToTensor(),
        owntransform.Normalize(*mean_std)
        ])

class TransformFix(object):
    def __init__(self, img_size,mean_std):

        self.weak = owntransform.Compose([
            owntransform.Resize((img_size, img_size)),
            owntransform.PadandRandomCrop(border=4, cropsize=(img_size, img_size)),
            owntransform.RandomHorizontalFlip(p=0.5)

        ])
        self.strong = owntransform.Compose([
            owntransform.Resize((img_size, img_size)),
            owntransform.PadandRandomCrop(border=4, cropsize=(img_size, img_size)),
            owntransform.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10)
            ])
        self.normalize = owntransform.Compose([
            owntransform.Resize((img_size, img_size)),
            owntransform.Normalize(*mean_std),
            owntransform.ToTensor()
           ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(x), self.normalize(weak), self.normalize(strong)

labeled_index=[]
labeled_targets=[]
def loading_data():
    transform_val = owntransform.Compose([
        owntransform.Resize((32, 32)),
        owntransform.Normalize(*mean_std),
        owntransform.ToTensor()
    ])
    if cfg.dataset == 'CIFAR10':

        root = './dataset/CIFAR10'
        base_dataset = datasets.CIFAR10( root, train=True, download=True)

        print("Dataset: CIFAR10")
        print(f"Unlabeled examples: {len(base_dataset.targets)-cfg.n_labeled}  after expand: {cfg.mu*cfg.n_imgs_per_epoch}")
        __, train_unlabeled_idxs = x_u_split(base_dataset.targets, cfg.n_labeled, num_classes=10)

        unlabeled_dataset = CIFAR10SSL(root, unlabeled_indexs=train_unlabeled_idxs, train=True,transform=TransformFix(32,mean_std))
        sampler_u = RandomSampler(unlabeled_dataset, replacement=True, num_samples=cfg.mu*cfg.n_imgs_per_epoch)
        batch_sampler_u = BatchSampler(sampler_u, cfg.mu*cfg.batch_size, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_sampler=batch_sampler_u, num_workers=4, pin_memory=False)

        test_dataset =CIFAR10SSL(root, train=False, transform=transform_val, download=False)
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size=1000,num_workers=4)

        return  unlabeled_loader, test_loader

    if cfg.dataset == 'CIFAR100':
        root = './dataset/CIFAR100'
        base_dataset = datasets.CIFAR100(root, train=True, download=True)

        print("Dataset: CIFAR100")
        print(f"Unlabeled examples: {len(base_dataset.targets) - cfg.n_labeled}  after expand: {cfg.mu * cfg.n_imgs_per_epoch}")
        __, train_unlabeled_idxs = x_u_split(base_dataset.targets, cfg.n_labeled, num_classes=100)

        unlabeled_dataset = CIFAR100SSL(root, unlabeled_indexs=train_unlabeled_idxs, train=True, transform=TransformFix(32,mean_std))
        sampler_u = RandomSampler(unlabeled_dataset, replacement=True, num_samples=cfg.mu*cfg.n_imgs_per_epoch)
        batch_sampler_u = BatchSampler(sampler_u, cfg.mu*cfg.batch_size, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_sampler=batch_sampler_u, num_workers=4, pin_memory=False)

        test_dataset =CIFAR100SSL(root, train=False, transform=transform_val, download=False)
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size=1000,num_workers=4)

        return unlabeled_loader, test_loader

    if cfg.dataset == 'STL10':
        root = './dataset/STL10'
        base_dataset = datasets.STL10(root, split='train', download=True)
        print("Dataset: STL10")
        print(f"Unlabeled examples: {len(base_dataset.labels) - cfg.n_labeled}  after expand: {cfg.mu * cfg.n_imgs_per_epoch}")
        __, train_unlabeled_idxs = x_u_split(base_dataset.labels,cfg.n_labeled, num_classes=10)

        unlabeled_dataset = STL10SSL(root, unlabeled_indexs=train_unlabeled_idxs, split='train', transform=TransformFix(48,mean_std))
        sampler_u = RandomSampler(unlabeled_dataset, replacement=True, num_samples=cfg.mu * cfg.n_imgs_per_epoch)
        batch_sampler_u = BatchSampler(sampler_u, cfg.mu * cfg.batch_size, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_sampler=batch_sampler_u, num_workers=4, pin_memory=False)

        test_dataset = STL10SSL(root, split='test', transform=transform_val, download=False)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=200, num_workers=8)

        return unlabeled_loader, test_loader

    if cfg.dataset == 'SVHN':

        root = './dataset/SVHN'
        base_dataset = datasets.SVHN(root, split='train', download=True)
        print("Dataset: SVHN")
        print(f"Unlabeled examples: {len(base_dataset.labels) - cfg.n_labeled}  after expand: {cfg.mu * cfg.n_imgs_per_epoch}")

        __, train_unlabeled_idxs = x_u_split(base_dataset.labels, cfg.n_labeled, num_classes=10)

        unlabeled_dataset = SVHNSSL(root, unlabeled_indexs=train_unlabeled_idxs, split='train', transform=TransformFix(32,mean_std))
        sampler_u = RandomSampler(unlabeled_dataset, replacement=True, num_samples=cfg.mu * cfg.n_imgs_per_epoch)
        batch_sampler_u = BatchSampler(sampler_u, cfg.mu * cfg.batch_size, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_sampler=batch_sampler_u, num_workers=4, pin_memory=False)

        test_dataset = SVHNSSL(root, split='test', transform=transform_val, download=False)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1000, num_workers=8)

        return unlabeled_loader,test_loader

    if cfg.dataset == 'MNIST':
        root = './dataset/MNIST'
        base_dataset = datasets.MNIST(root, train=True, download=True)

        print("Dataset: MNIST")
        print(f"Unlabeled examples: {len(base_dataset.targets) - cfg.n_labeled}  after expand: {cfg.mu * cfg.n_imgs_per_epoch}")
        __, train_unlabeled_idxs = x_u_split(base_dataset.targets, cfg.n_labeled, num_classes=10)

        unlabeled_dataset = MNISTSSL(root, unlabeled_indexs = train_unlabeled_idxs, train=True, transform=TransformFix(32, mean_std),download=True)
        sampler_u = RandomSampler(unlabeled_dataset, replacement=True, num_samples=cfg.mu*cfg.n_imgs_per_epoch)
        batch_sampler_u = BatchSampler(sampler_u, cfg.mu*cfg.batch_size, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_sampler=batch_sampler_u, num_workers=4, pin_memory=False)

        test_dataset =MNISTSSL(root, train=False, transform=transform_val, download=False)
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size=1000,num_workers=4)

        return unlabeled_loader, test_loader





def update_loading(indexs=None,pre_targets=None,epoch=None):
    if epoch ==0:
        all_indexs  = np.array(labeled_index)
        all_targets = labeled_targets

    else:
        labeled_index_app = np.array(indexs)
        labeled_index_app = labeled_index_app.flatten()  # 结果：array([1, 2, 3, 4])
        all_indexs = np.hstack([labeled_index_app,np.array(labeled_index)])
        all_targets = pre_targets+labeled_targets

    assert len(all_indexs) == len(all_targets)

    if cfg.dataset == 'CIFAR10':
        root = './dataset/CIFAR10'
        labeled_dataset = CIFAR10SSL(root, labeled_indexs=all_indexs, labeled_targets=all_targets, train=True, transform=TransformFix(32, mean_std))
        sampler_l = RandomSampler(labeled_dataset, replacement=True, num_samples=cfg.n_imgs_per_epoch)
        batch_sampler_l = BatchSampler(sampler_l, cfg.batch_size, drop_last=True)
        labeled_loader = DataLoader(labeled_dataset,   batch_sampler=batch_sampler_l, num_workers=4, pin_memory=False)

        print(f"Labeled examples: {len(all_indexs)}" f" after expand: {cfg.n_imgs_per_epoch}" )
        return labeled_loader

    if cfg.dataset == 'CIFAR100':
        root = './dataset/CIFAR100'
        labeled_dataset = CIFAR100SSL(root, labeled_indexs=all_indexs, labeled_targets=all_targets, train=True, transform=TransformFix(32, mean_std))
        sampler_l = RandomSampler(labeled_dataset, replacement=True, num_samples=cfg.n_imgs_per_epoch)
        batch_sampler_l = BatchSampler(sampler_l, cfg.batch_size, drop_last=True)
        labeled_loader = DataLoader(labeled_dataset, batch_sampler=batch_sampler_l, num_workers=4, pin_memory=False)
        print(f"Labeled examples: {len(all_indexs)}" f" after expand: {cfg.n_imgs_per_epoch}" )

        return labeled_loader

    if cfg.dataset == 'STL10':
        root = './dataset/STL10'
        labeled_dataset = STL10SSL(root, labeled_indexs=all_indexs, labeled_targets=all_targets, split='train', transform=TransformFix(48,mean_std))
        sampler_l = RandomSampler(labeled_dataset, replacement=True, num_samples=cfg.n_imgs_per_epoch)
        batch_sampler_l = BatchSampler(sampler_l, cfg.batch_size, drop_last=True)
        labeled_loader = DataLoader(labeled_dataset, batch_sampler=batch_sampler_l, num_workers=4, pin_memory=False)
        print(f"Labeled examples: {len(all_indexs)}" f" after expand: {cfg.n_imgs_per_epoch}" )

        return labeled_loader


    if cfg.dataset == 'MNIST':
        root = './dataset/MNIST'
        labeled_dataset = MNISTSSL(root,  labeled_indexs=all_indexs, labeled_targets=all_targets, train=True, transform=TransformFix(32, mean_std))
        sampler_l = RandomSampler(labeled_dataset, replacement=True, num_samples=cfg.n_imgs_per_epoch)
        batch_sampler_l = BatchSampler(sampler_l, cfg.batch_size, drop_last=True)
        labeled_loader = DataLoader(labeled_dataset, batch_sampler=batch_sampler_l, num_workers=4, pin_memory=False)
        print(f"Labeled examples: {len(all_indexs)}" f" after expand: {cfg.n_imgs_per_epoch}" )
        return labeled_loader

    if cfg.dataset == 'SVHN':
        root = './dataset/SVHN'
        labeled_dataset = SVHNSSL(root, labeled_indexs=all_indexs, labeled_targets=all_targets, split='train', transform=TransformFix(32, mean_std))
        sampler_l = RandomSampler(labeled_dataset, replacement=True, num_samples=cfg.n_imgs_per_epoch)
        batch_sampler_l = BatchSampler(sampler_l, cfg.batch_size, drop_last=True)
        labeled_loader = DataLoader(labeled_dataset, batch_sampler=batch_sampler_l, num_workers=4, pin_memory=False)
        print(f"Labeled examples: {len(all_indexs)}" f" after expand: {cfg.n_imgs_per_epoch}" )

        return labeled_loader


def x_u_split(labels,num_labeled, num_classes):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)

    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        unlabeled_idx.extend(idx[label_per_class:])
        labeled_targets.extend([i]*label_per_class)
        labeled_index.extend(idx[:label_per_class])

    return labeled_index, unlabeled_idx

if __name__ == "__main__":
    import random
    unlabel_loader, test_loader, = loading_data()

    print(len(unlabel_loader),len(test_loader))
    for i, ( a,b) in enumerate(zip(unlabel_loader,test_loader)):
        (ori_imgs, w_img, s_img), gt, index = a
        print(gt)
        imgs, label = b

        if i == 1:

            # print(imgs.size())
            # print(gt_map)
            # print(gt_map_b)
            # print(gt_map==gt_map_b)
            # print((gt_map==gt_map_b).sum())

            # print(index)

            img = torchvision.utils.make_grid(imgs).numpy()
            print(img.shape)
            img = np.transpose(
                img, (1, 2, 0)
            )  # \u5bf9\u539f\u6570\u7ec4\u7684\u8f6c\u7f6e,(0,1,2)\u5bf9\u5e94\u7740x,y,z\u8f74\u3002
            # img = img.transpose([1,2,0])
            print(img.shape)
            # img = np.resize(img, (544, 960*6))
            img = img[:, :, ::-1]
            print(img.shape)
            plt.imshow(img)
            plt.show()

            img = torchvision.utils.make_grid(ori_imgs).numpy()
            print(img.shape)
            img = np.transpose(
                img, (1, 2, 0)
            )  # \u5bf9\u539f\u6570\u7ec4\u7684\u8f6c\u7f6e,(0,1,2)\u5bf9\u5e94\u7740x,y,z\u8f74\u3002
            # img = img.transpose([1,2,0])
            print(img.shape)
            # img = np.resize(img, (544, 960*6))
            img = img[:, :, ::-1]
            print(img.shape)
            plt.imshow(img)
            plt.show()
        else:
            print(i,(gt_map == gt_map_b).sum())

    # train_loader, val_loader, shot_loader,train_dataset = loading_data()
    # for i, (imgs, gt_map,fake,index) in enumerate(shot_loader):
    #     if i == 1:
    #
    #         print(imgs.size(),gt_map,index)
    #
    #         img = torchvision.utils.make_grid(imgs).numpy()
    #         print(img.shape)
    #         img = np.transpose(
    #             img, (1, 2, 0)
    #         )  # \u5bf9\u539f\u6570\u7ec4\u7684\u8f6c\u7f6e,(0,1,2)\u5bf9\u5e94\u7740x,y,z\u8f74\u3002
    #         # img = img.transpose([1,2,0])
    #         print(img.shape)
    #         # img = np.resize(img, (544, 960*6))
    #         img = img[:, :, ::-1]
    #         print(img.shape)
    #         plt.imshow(img)
    #         plt.show()
    #         break