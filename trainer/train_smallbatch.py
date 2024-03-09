import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_manager import VCM_Pose
from dataset_preparation import PoseDataset_train
from demo import GenIdx
# from net_lstm_withmore import DualStreamPoseNet as net
from net.EnhancedLstm import PoseFeatureNet as net
from util import IdentitySampler

if __name__ == '__main__':
    """这个文件是为了验证网络是否能够正常训练
    原理是：利用一个小的数据集，重复训练多次，看看网络是否能够快速收敛
    收敛的标准是：损失函数的值是否能够快速下降趋向于0
    """
    pose = VCM_Pose()
    rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)
    sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, 2, 8)
    pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=12, sample='random', transform=None, index1=sampler.index1, index2=sampler.index2)
    dataloader = DataLoader(pose_dataset, batch_size=8, num_workers=0, drop_last=True)

    criterion = nn.CrossEntropyLoss().cuda()
    net = net(500).cuda()
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.09, weight_decay=5e-4)
    # 提取数据
    saved_batches = []
    # losses = []
    for batch_idx, data in enumerate(dataloader):
        if batch_idx < 50:  # 只保存前2个批次的数据
            saved_batches.append(data)
        else:
            break

    # 重复学习
    num_epochs = 500  # 重复训练的轮数
    for epoch in range(num_epochs):
        total_loss = 0
        for data in saved_batches:
            imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb = data
            input_rgb = Variable(imgs_rgb.cuda().float())
            input_ir = Variable(imgs_ir.cuda().float())
            label1 = Variable(pids_rgb.cuda())
            label2 = Variable(pids_ir.cuda())
            feature, feature_cls = net(input_rgb, input_ir)
            # loss = criterion(rgb_feature, label1) + criterion(ir_feature, label2)
            label = torch.cat((pids_rgb, pids_ir), 0)
            labels = Variable(label.cuda())
            loss = criterion(feature_cls, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # losses.append(total_loss / len(saved_batches))
        # clear_output(wait=True)  # 清除之前的输出
        # plt.plot(losses, label='Training Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()
        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(saved_batches)}")
