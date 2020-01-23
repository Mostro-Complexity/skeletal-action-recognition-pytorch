import json
import os

import cv2
import h5py
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, Resize


class MSRAction3D(torch.utils.data.Dataset):
    def __init__(self, root, method, split_id=0, resize_isize=(60, 60, 3)):
        super(MSRAction3D, self).__init__()
        self.method = method
        self.split_id = split_id
        self.resize_isize = resize_isize
        # 读取特征
        features = self.load_features(os.path.join(root, 'features.mat'))
        # 读取标签
        subject_labels, action_labels = self.load_labels(
            os.path.join(root, 'labels.mat'))

        # 读取数据集划分方案
        tr_subjects, te_subjects, n_tr_te_splits = self.load_splits_config(
            os.path.join(root, 'tr_te_splits.json'))

        self.body_model = json.load(
            open(os.path.join(root, 'body_model.json'), 'r'))

        if method == 'train':
            features, labels = self.split_dataset(
                features, action_labels, subject_labels, tr_subjects[split_id, :])
        elif method == 'test':
            features, labels = self.split_dataset(
                features, action_labels, subject_labels, te_subjects[split_id, :])

        self.n_features = features.shape[0]

        self.features = torch.tensor(features)
        self.labels = torch.tensor(labels-1).long()
        # self.labels = torch.tensor(to_categorical(labels-1))

    def load_features(self, path):
        f = h5py.File(path, 'r')

        features = [f[element] for element in np.squeeze(f['features'][:])]

        features = np.array(features)
        return features

    def load_labels(self, path):
        f = h5py.File(path, 'r')

        subject_labels = np.array(f['subject_labels'][:, 0])
        action_labels = np.array(f['action_labels'][:, 0])
        return subject_labels, action_labels

    def load_splits_config(self, path):
        f = json.load(open(path, 'r'))

        tr_subjects = np.array(f['tr_subjects'])
        te_subjects = np.array(f['te_subjects'])
        n_tr_te_splits = tr_subjects.shape[0]

        return tr_subjects, te_subjects, n_tr_te_splits

    def split_dataset(self, features, action_labels, subject_labels, subjects):
        subject_ind = np.isin(subject_labels, subjects)
        labels = action_labels[subject_ind]

        features = features[subject_ind]
        return features, labels

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        return feature, label

    def __len__(self):
        return self.n_features

