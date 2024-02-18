import numpy as np

from dataset_preparation import PoseDataset
from torch.utils.data import DataLoader
from data_manager import VCM_Pose
from util import IdentitySampler


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos


if __name__ == '__main__':
    pose = VCM_Pose()

    rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)

    sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, 2, 4)
    index1 = sampler.index1
    index2 = sampler.index2

    pose_dataset = PoseDataset(pose.train_ir, pose.train_rgb, seq_len=12, sample='random', transform=None,
                               index1=index1, index2=index2)

    dataloader = DataLoader(pose_dataset, batch_size=4, num_workers=1, drop_last=True, )

    for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(dataloader):
        print("--------------------------------------------------------------------------------")
        print(f"Batch Index: {batch_idx}")
        print("Infrared Data - Images: {}, Person IDs: {}, Camera IDs: {}".format(imgs_ir.shape, pids_ir, camid_ir))
        print("RGB Data - Images: {}, Person IDs: {}, Camera IDs: {}".format(imgs_rgb.shape, pids_rgb, camid_rgb))
        # 这里的imgs_ir和imgs_rgb是一个batch的数据，形状是[4, 228, 6]
        # 其中4是batch_size，228是19*12，19是人物关节数量，12是序列长度，6是特征的六个维度（两对关键点，一个欧氏距离，一个角度）
        break
