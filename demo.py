import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from dataset_preparation import PoseDataset_train, PoseDataset_test
from torch.utils.data import DataLoader
from data_manager import VCM_Pose
from util import IdentitySampler, evaluate
from demo_net import PoseFeatureNet
from net_lstm import PoseFeatureNet as net



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

def test():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    ptr = 0
    gall_feat = np.zeros((ngall, 500))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            input = imgs
            label = pids
            batch_num = input.size(0)

            input = Variable(input.float().cuda())
            feat = net(input, input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            g_pids.extend(pids)
            g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat = np.zeros((nquery, 500))

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            input = imgs
            label = pids

            batch_num = input.size(0)

            input = Variable(input.float().cuda())
            feat = net(input, input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP

def test1():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader_1):
            input = imgs
            label = pids
            batch_num = input.size(0)

            input = Variable(input.float().cuda())
            feat = net(input, input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            g_pids.extend(pids)
            g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat = np.zeros((nquery, 2048))

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader_1):
            input = imgs
            label = pids

            batch_num = input.size(0)

            input = Variable(input.float().cuda())
            feat = net(input, input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP
if __name__ == '__main__':
    pose = VCM_Pose()

    # TODO: 这里的采样器的作用是什么，以及下面的sanple有什么不同
    rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)

    sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, 2, 1)
    index1 = sampler.index1
    index2 = sampler.index2

    pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=12, sample='random', transform=None,
                                     index1=index1, index2=index2)

    dataloader = DataLoader(pose_dataset, batch_size=1, num_workers=16, drop_last=True, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda')

    net = net(500)
    net.to('cuda')

    nquery_1 = pose.num_query_tracklets_1
    ngall_1 = pose.num_gallery_tracklets_1
    nquery = pose.num_query_tracklets
    ngall = pose.num_gallery_tracklets

    queryloader = DataLoader(
        PoseDataset_test(pose.query, seq_len=12, sample='video_test'),
        batch_size=32, shuffle=False, num_workers=1)

    galleryloader = DataLoader(
        PoseDataset_test(pose.gallery, seq_len=12, sample='video_test'),
        batch_size=32, shuffle=False, num_workers=1)
    # ----------------visible to infrared----------------
    queryloader_1 = DataLoader(
        PoseDataset_test(pose.query_1, seq_len=12, sample='video_test'),
        batch_size=32, shuffle=False, num_workers=1)

    galleryloader_1 = DataLoader(
        PoseDataset_test(pose.gallery_1, seq_len=12, sample='video_test'),
        batch_size=32, shuffle=False, num_workers=1)
    # optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(dataloader):
        print("--------------------------------------------------------------------------------")
        print(f"Batch Index: {batch_idx}")
        # 这里的imgs_ir和imgs_rgb是一个batch的数据，形状是[4, 12, 19, 6]
        # 其中4是batch_size，12是序列长度,19是人物关节数量，6是特征的六个维度（两对关键点，一个欧氏距离，一个角度）
        print("Infrared Data - Images: {}, Person IDs: {}, Camera IDs: {}".format(imgs_ir.shape, pids_ir, camid_ir))
        print("RGB Data - Images: {}, Person IDs: {}, Camera IDs: {}".format(imgs_rgb.shape, pids_rgb, camid_rgb))

        # 将数据放入网络：
        input_rgb = Variable(imgs_rgb.cuda().float())
        input_ir = Variable(imgs_ir.cuda().float())

        label1 = pids_rgb
        label2 = pids_ir

        label1 = Variable(label1.cuda())
        label2 = Variable(label2.cuda())

        rgb_feature, ir_feature = net(input_rgb, input_ir)

        # print("RGB Feature: ", rgb_feature)
        # print("RGB Label: ", label1)

        # 计算损失
        loss = criterion(rgb_feature, label1) + criterion(ir_feature, label2)

        print("Loss: ", loss.item())

        # 梯度清零
        optimizer.zero_grad()
        # 优化器的更新
        loss.backward()

        # 参数更新
        optimizer.step()

        # print("--------------------------------------------------------------------------------")

    cmc, mAP = test()
    print(
        'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP))

    cmc, mAP = test1()
    print(
        'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP))
