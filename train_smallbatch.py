import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# sys.path.append("..")
from data_manager import VCM_Pose
from dataset_preparation import PoseDataset_train
from util import GenIdx
from net.st_gcn.st_gcn import Model as net
from util import IdentitySampler


if __name__ == '__main__':
    """这个文件是为了验证网络是否能够正常训练
    原理是：利用一个小的数据集，重复训练多次，看看网络是否能够快速收敛
    收敛的标准是：损失函数的值是否能够快速下降趋向于0
    """
    pose = VCM_Pose()
    rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)
    sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, 2, 8)
    pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=12, sample='video_train', transform=None, index1=sampler.index1, index2=sampler.index2)
    dataloader = DataLoader(pose_dataset, batch_size=8, num_workers=2, drop_last=True)

    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = torch.nn.MSELoss()
    net = net(in_channels=3,
              num_class=500,
              graph_args={'layout': 'coco', 'strategy': 'spatial'},
              edge_importance_weighting=True).cuda()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.09, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    saved_batches = []
    for batch_idx, data in enumerate(dataloader):
        if batch_idx < 4:  # 只保存前2个批次的数据
            saved_batches.append(data)
        else:
            break

    num_epochs = 50
    losses = []  # 用于记录每个epoch的平均损失
    for epoch in range(num_epochs):
        total_loss = 0
        for data in saved_batches:
            imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb = data

            input1 = imgs_rgb
            input2 = imgs_ir
            label1 = pids_rgb
            label2 = pids_ir

            input_rgb = Variable(input1.float().to('cuda').permute(0, 3, 1, 2).unsqueeze(-1))
            input_ir = Variable(input2.float().to('cuda').permute(0, 3, 1, 2).unsqueeze(-1))

            labels = torch.cat((label1, label2), 0)  # 将两个模态的标签拼接起来，形成一个新的标签
            labels = Variable(labels.cuda())

            # concat_feature, feature_cls = net(input_rgb, input_ir)
            feature_rgb = net(input_rgb)
            feature_ir = net(input_ir)

            feature_cls = torch.cat((feature_rgb, feature_ir), 0)
            loss = criterion1(feature_cls, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(saved_batches)
        losses.append(avg_loss)  # 将平均损失添加到列表中
        print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
        # if epoch==20:
        #     save_model(net, 'starter.pth')
