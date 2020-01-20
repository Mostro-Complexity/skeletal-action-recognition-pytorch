import json
import os
from abc import ABCMeta, abstractmethod

import cv2
import h5py
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, Resize


class PreprocessPipeline(metaclass=ABCMeta):
    @abstractmethod
    def skeleton2img(self, mat):
        pass

    @abstractmethod
    def standardize_pose(self, mat, resize_isize):
        pass


class MSRAction3D(torch.utils.data.Dataset, PreprocessPipeline):
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
        feature = self.skeleton2img(self.features[index])
        feature = self.standardize_pose(feature, self.resize_isize)
        feature = feature.permute(2, 0, 1)
        label = self.labels[index]

        return feature, label

    def __len__(self):
        return self.n_features

    def skeleton2img(self, mat):
        '''map skeletal action to rgb image
        :param mat: (n_frames, feat_dim)
        :return: image with resize_isize
        '''
        # mat = np.reshape(mat, newshape=(-1, 20, 3))
        mat = mat.transpose(0, 1)  # mat: (n_feat, n_frames, n_dim)

        n_frames = mat.shape[1]
        # part_config = [25, 12, 24, 11, 10, 9, 21, 21, 5, 6, 7, 8, 22, 23,
        #                21, 3, 4, 21, 2, 1, 17, 18, 19, 20, 21, 2, 1, 13, 14, 15, 16]
        part_config = torch.tensor([12, 10, 8, 1, 1, 3, 2, 2, 9, 11, 13, 20, 3, 4, 7, 7,
                                    5, 6, 5, 14, 16, 18, 6, 15, 17, 19])
        mat = mat[part_config - 1]

        return mat

    def standardize_pose(self, mat, resize_isize):
        '''standardize each pixel of this image
        '''
        local_max = torch.max(mat)
        local_min = torch.min(mat)
        mat = (mat-local_min)/(local_max-local_min)

        rgb_image = cv2.resize(mat.numpy(), resize_isize[:2])
        # cv2.imshow('123',rgb_image)
        # cv2.waitKey()
        return torch.tensor(rgb_image)
