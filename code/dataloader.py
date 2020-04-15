#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#===============================================================================
''' Dataset initialization & get dataloader '''
class CIFAR10_Dataloader():
    def __init__(self, path='../dataset/CIFAR10/'):
        # Set dataset path
        self.path = path
        # Set labels of 10 classes
        self.classes = ('plane', 'car', 'bird', 'cat','deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        # Set dataset transform
        self.transform_train = transforms.Compose([                              #   <- 필요없는 변환 지우기
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.RandomResizedCrop(32, scale=(0.08, 1.0),
                        #             ratio=(0.75, 1.3333), interpolation=2),
                        # transforms.RandomRotation(10, resample=False,
                        #             expand=False, center=None),
                        # transforms.ColorJitter(brightness=0.4, contrast=0.4,
                        #             saturation=0.4, hue=0),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.247, 0.243, 0.261)),
                        ])
        self.transform_eval = transforms.Compose([
                        # transforms.CenterCrop(32)
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.247, 0.243, 0.261)),
                        ])
        # Better normalization value than 0.5
        # Reference: https://github.com/kuangliu/pytorch-cifar/issues/19

    ''' Get train dataset loader '''
    def get_train_loader(self, batch_size=32, num_workers=2):
        trainset = torchvision.datasets.CIFAR10(root=self.path,
                                                train=True,
                                                download=True,
                                                transform=self.transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
        return train_loader

    ''' Get eval dataset loader '''
    def get_val_loader(self, batch_size=32, num_workers=2):
        valset = torchvision.datasets.CIFAR10(root=self.path,
                                                train=False,
                                                download=True,
                                                transform=self.transform_eval)
        val_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
        return val_loader

#===============================================================================
''' Test dataloader '''
if __name__ == "__main__":
    # Get images and labels from dataloader
    cifar10 = CIFAR10_Dataloader()
    dataloader = cifar10.get_train_loader(batch_size=8, num_workers=6)
    # dataloader = cifar10.get_eval(batch_size=8, num_workers=6)
    images, labels = iter(dataloader).next()

    # Show images and labels
    images = torchvision.utils.make_grid(images)
    images = images/2+0.5       # Un-Normalize
    labels = ' '.join('%5s' % cifar10.classes[labels[ii]] for ii in range(8))
    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
    plt.xlabel(labels)
    plt.show()
