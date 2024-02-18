import math
import random

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb

        self.seq_len = seq_len
        self.sample = sample

        self.transform = transform

        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)

        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len  # 这个地方的S是指的seq_len

        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))
        sample_clip_ir = np.array(sample_clip_ir)

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))
        sample_clip_rgb = np.array(sample_clip_rgb)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_ir_paths[index]
                img = np.load(img_path)
                img = torch.tensor(img)
                # img = np.array(img)
                # if self.transform is not None:
                #     img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            frame_indices = range(num_rgb)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break

                indices.append(index)
            indices = np.array(indices)
            imgs_rgb = []
            for index in indices:
                index = int(index)
                img_path = img_rgb_paths[index]
                img = np.load(img_path)
                img = torch.tensor(img)
                # img = np.array(img)
                # if self.transform is not None:
                #     img = self.transform(img)

                imgs_rgb.append(img)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb

        elif self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

            imgs_ir = []
            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img = np.load(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []
            for index in number_rgb:
                index = int(index)
                img_path = img_rgb_paths[index]
                img = np.load(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_rgb.append(img)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb  # 返回了两种模态的三种信息，分别是图像、标签和摄像头ID。

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
