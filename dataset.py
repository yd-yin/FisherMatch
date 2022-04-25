import os
from os.path import join
import numpy as np
from lib.Dataset_Base import Dataset_Base
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d import transforms as trans
from torchvision import transforms
from dataset_pascal import get_dataloader_pascal3d


class ModelNetDataset(Dataset_Base):
    def __init__(self, data_dir, category, collection='train', net_arch='alexnet', sample_inds=None, aug=None, aug_strong=None):
        super(ModelNetDataset, self).__init__(data_dir, category, collection, net_arch, 1.0, sample_inds)
        self.aug = aug
        self.aug_strong = aug_strong

    def __getitem__(self, idx):
        rc = self.recs[idx]
        cate = rc.category
        img_id = rc.img_id
        quat = rc.so3.quaternion
        quat = torch.from_numpy(quat)
        rot_mat = trans.quaternion_to_matrix(quat)

        img = self._get_image(rc)
        img = torch.from_numpy(img)

        if self.aug is not None:
            img = self.aug(img)

        if self.aug_strong is not None:
            img_strong = self.aug_strong(img)
        else:
            img_strong = torch.zeros_like(img)

        sample = dict(
            idx=idx,
            label=self.cate2ind[cate],
            quat=quat,
            rot_mat=rot_mat,
            img=img,
            img_strong=img_strong,
            img_id=img_id
        )

        return sample



def get_dataloader(dataset, phase, config):
    if dataset == 'modelnet':
        return get_dataloader_modelnet(phase, config)
    elif dataset == 'pascal3d':
        return get_dataloader_pascal3d(phase, config)


def get_dataloader_modelnet(phase, config):
    weak_augment = transforms.Compose([
        transforms.Pad((20, 20), padding_mode='edge'),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.), ratio=(1., 1.))
    ])
    strong_augment = transforms.Compose([
        transforms.Pad((60, 60), padding_mode='edge'),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.), ratio=(1., 1.))
    ])

    if phase == 'train':
        if config.ss_ratio < 1.:
            sample_inds = np.load(join(config.data_dir, 'ModelNet10-SO3', f'train_100V_{config.category}_r{config.ss_ratio}.npy'))
        else:
            sample_inds = None
        batch_size = config.batch_size
        collection = 'train'
        shuffle = True
        if config.ss_ratio < 1.:
            aug = weak_augment
        else:
            aug = None
        aug_strong = None

    elif phase == 'ulb_train':
        sample_inds = np.load(join(config.data_dir, 'ModelNet10-SO3', f'train_100V_{config.category}_r{1 - config.ss_ratio}.npy'))
        batch_size = round(config.batch_size * config.ulb_batch_ratio)
        collection = 'train'
        shuffle = True
        aug = weak_augment
        aug_strong = strong_augment

    elif phase == 'test':
        sample_inds = None
        batch_size = config.batch_size
        collection = 'test'
        shuffle = False
        aug = None
        aug_strong = None

    else:
        raise ValueError

    dset = ModelNetDataset(config.data_dir, config.category, collection=collection, net_arch='vgg16', sample_inds=sample_inds, aug=aug, aug_strong=aug_strong)

    dloader = DataLoader(dset, batch_size=batch_size, num_workers=config.num_workers, shuffle=shuffle, pin_memory=True)

    return dloader

