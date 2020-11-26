import numpy as np
from torchvision import datasets


class STL10SSL(datasets.STL10):
    def __init__(self, root, unlabeled_indexs=None, labeled_indexs=None,labeled_targets=None,split=None,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.unlabeled_indexs =unlabeled_indexs
        if unlabeled_indexs is not None:
            self.data = self.data[unlabeled_indexs]
            self.data_id = unlabeled_indexs
            self.targets = np.array(self.labels)[unlabeled_indexs]

        if  (labeled_indexs is not None) and  (labeled_targets is not None):
            real = np.array(self.labels)[labeled_indexs]
            self.data = self.data[labeled_indexs]
            self.targets = np.array(labeled_targets)
            print(f'acc:  {(self.targets==real).sum()/len(self.targets) * 100}%')

        if (unlabeled_indexs is None) and (labeled_indexs is None) and (labeled_targets is None):
            self.targets =self.labels

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = np.transpose(img, (1, 2, 0))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.unlabeled_indexs is not None:
            return img, target,self.data_id[index]
        else:
            return img, target

