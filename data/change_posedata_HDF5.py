import numpy as np
import h5py
import os

from tqdm import tqdm

# 定义.npy文件所在的目录
npy_dir = 'VCM-POSE/Test'

# 定义输出的HDF5文件路径
hdf5_path = 'VCM-POSE-HDF5-Test.hdf5'

# 创建HDF5文件
with h5py.File(hdf5_path, 'w') as hdf5_file:
    # 遍历.npy文件所在的目录
    for root, dirs, files in os.walk(npy_dir):
        for file in tqdm(files):
            if file.endswith('.npy'):
                # 构造.npy文件的完整路径
                npy_path = os.path.join(root, file)

                # 使用文件的相对路径作为HDF5中的键
                relative_path = os.path.relpath(npy_path, npy_dir)
                # 将斜线转换为下划线，如果你的文件路径作为键名时需要
                hdf5_key = relative_path.replace(os.sep, '_')

                # 读取.npy文件内容
                data = np.load(npy_path)

                # 将数据写入HDF5文件
                hdf5_file.create_dataset(hdf5_key, data=data)

print("数据集转换完成。")

