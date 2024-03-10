import math
import random

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
class PoseDataset_train(Dataset):
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb

        self.seq_len = seq_len
        self.sample = sample

        self.transform = transform

        self.index1 = index1
        self.index2 = index2
        # self.hdf5_file = h5py.File('data/VCM-POSE-HDF5-Train.hdf5', 'r')

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)

        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len  # 这个地方的S是指的seq_len

        hdf5_file = h5py.File('../data/VCM-POSE-HDF5-Train.hdf5', 'r')

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
                img_path = img_path[15:]
                img_hdf5_key = img_path.replace('/', '_')
                img = hdf5_file[img_hdf5_key][()]
                imgs_ir.append(img)
            # imgs_ir = torch.cat(imgs_ir, dim=0)
            # imgs_ir = torch.stack(imgs_ir, dim=0)
            imgs_ir_np = np.stack(imgs_ir, axis=0)
            imgs_ir_tensor = torch.from_numpy(imgs_ir_np).float()

            # ----------------------------------------------------
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
                img_path = img_path[15:]
                img_hdf5_key = img_path.replace('/', '_')
                img = hdf5_file[img_hdf5_key][()]
                imgs_rgb.append(img)
            imgs_rgb_np = np.stack(imgs_rgb, axis=0)
            imgs_rgb_tensor = torch.from_numpy(imgs_rgb_np).float()
            # imgs_rgb = torch.stack(imgs_rgb, dim=0)

            return imgs_ir_tensor, pid_ir, camid_ir, imgs_rgb_tensor, pid_rgb, camid_rgb

        elif self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]
            imgs_ir = []
            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img_path = img_path[23:]
                img_hdf5_key = img_path.replace('/', '_').replace('.jpg', '.npy')
                img = hdf5_file[img_hdf5_key][()]
                imgs_ir.append(img)
            imgs_ir_np = np.stack(imgs_ir, axis=0)
            imgs_ir_tensor = torch.from_numpy(imgs_ir_np).float()

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []
            for index in number_rgb:
                index = int(index)
                img_path = img_rgb_paths[index]
                img_path = img_path[23:]
                img_hdf5_key = img_path.replace('/', '_').replace('.jpg', '.npy')
                img = hdf5_file[img_hdf5_key][()]
                imgs_rgb.append(img)
            imgs_rgb_np = np.stack(imgs_rgb, axis=0)
            imgs_rgb_tensor = torch.from_numpy(imgs_rgb_np).float()
            return imgs_ir_tensor, pid_ir, camid_ir, imgs_rgb_tensor, pid_rgb, camid_rgb  # 返回了两种模态的三种信息，分别是图像、标签和摄像头ID。

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class PoseDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        # 定义了数据集、序列长度、采样策略和数据变换等参数。
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        # self.hdf5_file = h5py.File('data/VCM-POSE-HDF5-Test.hdf5', 'r')

    def __len__(self):
        # 返回数据集中样本的个数
        return len(self.dataset)

    def __getitem__(self, index):
        # 根据索引获取一个样本
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))

        hdf5_file = h5py.File('../data/VCM-POSE-HDF5-Test.hdf5', 'r')

        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)
        # 代码中实现了三种采样策略：evenly、random 和 all。
        # 'dense'：将每个视频样本分成长度为 seq_len 的若干个连续视频片段，并返回所有片段的样本。
        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        # 'random'：在每个视频样本中随机抽取长度为 seq_len 的连续视频片段，并返回抽取的片段的样本。
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
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
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid  # 这里返回了一个视频片段的样本 imgs_ir，以及该样本对应的行人 ID 和摄像头 ID。

        # 'video_test'：使用预先计算好的视频片段采样方案，从每个视频样本中抽取相应的样本。
        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img_path = img_path[22:]
                img_hdf5_key = img_path.replace('/', '_').replace('.jpg', '.npy')
                img = hdf5_file[img_hdf5_key][()]
                # img = np.load(img_path)
                # img = torch.tensor(img)
                # img = torch.from_numpy(img)
                # img = np.array(img)
                # if self.transform is not None:
                #     img = self.transform(img)

                imgs_ir.append(img)
            # imgs_ir = torch.cat(imgs_ir, dim=0)
            # imgs_ir = torch.stack(imgs_ir, dim=0)
            imgs_ir_np = np.stack(imgs_ir, axis=0)
            imgs_ir_tensor = torch.from_numpy(imgs_ir_np).float()
            return imgs_ir_tensor, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
