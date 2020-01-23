import os

import cv2
import h5py
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, Resize


class MappingPipeline(torch.utils.data.Dataset):
    def __init__(self, dataset, resize_isize=(60, 60, 3)):
        super(MappingPipeline, self).__init__()
        self.resize_isize = resize_isize
        self.dataset = dataset

    def __getitem__(self, index):
        feature = self.skeleton2img(self.dataset.features[index])
        feature = self.standardize_pose(feature, self.resize_isize)
        feature = feature.permute(2, 0, 1)
        label = self.dataset.labels[index]

        return feature, label

    def __len__(self):
        return self.dataset.n_features

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


class MultiMappingPipeline(MappingPipeline):
    def __init__(self, dataset, resize_isize=(60, 60, 3),
                 method='all'):
        super(MultiMappingPipeline, self).__init__(
            dataset=dataset, resize_isize=resize_isize)
        self.method = method

    def __getitem__(self, index):
        ret_pack = []
        if self.method == 'abs_pos' or self.method == 'all':
            abs_pos = self.skeleton2img(self.dataset.features[index])
            abs_pos = self.standardize_pose(abs_pos, self.resize_isize)
            abs_pos = abs_pos.permute(2, 0, 1)
            ret_pack.append(abs_pos)
        if self.method == 'lp_ang' or self.method == 'all':
            lp_ang = self.compute_cross_angle(
                self.dataset.features[index],
                self.dataset.body_model['lp_angle_pairs'])
            lp_ang = self.standardize_image(lp_ang)
            lp_ang = lp_ang.permute(2, 0, 1)
            ret_pack.append(lp_ang)
        if self.method == 'll_ang' or self.method == 'all':
            ll_ang = self.compute_dot_angle(
                self.dataset.features[index],
                self.dataset.body_model['ll_angle_pairs'])
            ll_ang = self.standardize_image(ll_ang)
            ll_ang = ll_ang[None, :, :]
            ret_pack.append(ll_ang)

        label = self.dataset.labels[index]
        ret_pack.append(label)
        return tuple(ret_pack)

    def standardize_image(self, mat):
        mat = mat.transpose(0, 1)  # mat: (n_feat, n_frames, n_dim)
        n_frames = mat.size(1)

        mat = self.standardize_pose(mat, self.resize_isize)
        return mat

    def compute_cross_angle(self, mat, lp_angle_pairs):
        '''
        :mat: absolute joint locations
        :lp_angle_pairs: indices of start points and end points of angles
        '''
        lp_angle_pairs = torch.tensor(lp_angle_pairs) - 1
        # lp_angle_pairs = lp_angle_pairs[::2]

        n_pairs = lp_angle_pairs.size(0)
        n_frames, _, n_dim = mat.size()

        cross = torch.empty((n_frames, n_pairs, n_dim))

        for t in range(n_frames):
            rls_1 = mat[t, lp_angle_pairs[:, 1]]-mat[t, lp_angle_pairs[:, 0]]
            rls_2 = mat[t, lp_angle_pairs[:, 2]]-mat[t, lp_angle_pairs[:, 0]]
            rls_0 = mat[t, lp_angle_pairs[:, 2]]-mat[t, lp_angle_pairs[:, 1]]
            cross[t] = torch.cross(rls_1, rls_2, dim=1)
            norm = torch.norm(rls_0, p=2, dim=1)
            cross[t] /= norm[:, np.newaxis]

        return cross

    def compute_dot_angle(self, mat, ll_angle_pairs):
        '''
        :mat: absolute joint locations
        :ll_angle_pairs: indices of start points and end points of angles
        '''
        ll_angle_pairs = torch.tensor(ll_angle_pairs) - 1
        ll_angle_pairs = ll_angle_pairs[::2]
        n_pairs = ll_angle_pairs.size(0)
        n_frames, _, n_dim = mat.size()

        dot = torch.empty(n_frames, n_pairs)

        for t in range(n_frames):
            rls_1 = mat[t, ll_angle_pairs[:, 1]]-mat[t, ll_angle_pairs[:, 0]]
            rls_1 /= torch.norm(rls_1, p=2, dim=1)[:, None]
            rls_2 = mat[t, ll_angle_pairs[:, 3]]-mat[t, ll_angle_pairs[:, 2]]
            rls_2 /= torch.norm(rls_2, p=2, dim=1)[:, None]

            dot[t] = torch.sum(rls_1*rls_2, dim=1)
            dot_frame = dot[t]
            dot_frame[dot_frame > 1] = 1
            dot_frame[dot_frame < -1] = -1
            dot[t] = torch.acos(dot_frame)
        return dot
