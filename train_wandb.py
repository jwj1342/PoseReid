import time

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_manager import VCM_Pose
from dataset_preparation import PoseDataset_train, PoseDataset_test
# from net_lstm import PoseFeatureNet as net
from net_lstm_withmore import DualStreamPoseNet as net
from util import IdentitySampler, evaluate
import wandb


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


def extract_features_no_grad(data_loader, feature_dimension, net):
    """提取特征的通用函数"""
    features = np.zeros((len(data_loader.dataset), feature_dimension))
    pids = []
    camids = []
    ptr = 0
    with torch.no_grad():
        for batch_idx, (imgs, pids_batch, camids_batch) in enumerate(data_loader):
            input_imgs = Variable(imgs.float().cuda())
            batch_num = input_imgs.size(0)
            feats = net(input_imgs, input_imgs)
            features[ptr:ptr + batch_num, :] = feats.detach().cpu().numpy()
            ptr += batch_num
            pids.extend(pids_batch)
            camids.extend(camids_batch)
    return features, np.asarray(pids), np.asarray(camids)


def test_general(gallery_loader, query_loader, net, ngall, nquery):
    net.eval()

    print('Extracting Gallery Feature...')
    gall_feat, g_pids, g_camids = extract_features_no_grad(gallery_loader, 500, net)

    print('Extracting Query Feature...')
    query_feat, q_pids, q_camids = extract_features_no_grad(query_loader, 500, net)

    # 计算相似度
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


def save_model(net, path):
    """保存模型参数"""
    torch.save(net.state_dict(), path)


def load_model(net, path):
    """加载模型参数"""
    net.load_state_dict(torch.load(path))


if __name__ == '__main__':
    pose = VCM_Pose()

    rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)

    sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, 2, 128)
    index1 = sampler.index1
    index2 = sampler.index2

    pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=12, sample='random', transform=None,
                                     index1=index1, index2=index2)

    dataloader = DataLoader(pose_dataset, batch_size=128, num_workers=0, drop_last=True, sampler=sampler)

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
        batch_size=64, shuffle=False, num_workers=0)

    galleryloader = DataLoader(
        PoseDataset_test(pose.gallery, seq_len=12, sample='video_test'),
        batch_size=64, shuffle=False, num_workers=0)
    # ----------------visible to infrared----------------
    queryloader_1 = DataLoader(
        PoseDataset_test(pose.query_1, seq_len=12, sample='video_test'),
        batch_size=64, shuffle=False, num_workers=0)

    galleryloader_1 = DataLoader(
        PoseDataset_test(pose.gallery_1, seq_len=12, sample='video_test'),
        batch_size=64, shuffle=False, num_workers=0)

    wandb.init(project="train_pose_LSTM_DualStream",  # track hyperparameters and run metadata
               config={
                   "optimizer": "Adam",
                   "learning_rate": 0.001,
                   "architecture": "LSTM DualStream",
                   "dataset": "VCM-POSE",
                   "epochs": 81,
               })
    wandb.watch(net, log="all", log_freq=10)

    # optimizer = optim.SGD(net.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    best_mAP = 0
    load_model(net, "best_model_lstmaddon.pth")
    for epoch in range(wandb.config.epochs):
        running_loss = 0.0
        start_time = time.time()  # 开始时间
        net.train()
        for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(dataloader):
            input_rgb = Variable(imgs_rgb.float().to('cuda'))
            input_ir = Variable(imgs_ir.float().to('cuda'))

            label1 = Variable(pids_rgb.to('cuda'))
            label2 = Variable(pids_ir.to('cuda'))

            rgb_feature, ir_feature = net(input_rgb, input_ir)

            loss = criterion(rgb_feature, label1) + criterion(ir_feature, label2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()

        avg_loss = running_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch:{epoch}  Loss:{avg_loss} Time: {end_time - start_time} s")

        if epoch % 10 == 0:
            save_model(net, "best_model.pth")
            cmc_t2v, mAP_t2v = test_general(galleryloader, queryloader, net, ngall, nquery)
            wandb.log({"epoch": epoch, "mAP_t2v": mAP_t2v})

            cmc_v2t, mAP_v2t = test_general(galleryloader_1, queryloader_1, net, ngall_1, nquery_1)
            wandb.log({"epoch": epoch, "mAP_v2t": mAP_v2t})
        # 每个epoch结束后进行测试
        # start_time = time.time()
        # cmc_t2v, mAP_t2v = test_general(galleryloader, queryloader, net, ngall, nquery)
        # end_time = time.time()
        # print(f"t2v_Test_Time: {end_time - start_time} s")
        # wandb.log({"epoch": epoch, "mAP_t2v": mAP_t2v})
        #
        # start_time = time.time()
        # cmc_v2t, mAP_v2t = test_general(galleryloader_1, queryloader_1, net, ngall_1, nquery_1)
        # end_time = time.time()
        # print(f"v2t_Test_Time: {end_time - start_time} s")
        # wandb.log({"epoch": epoch, "mAP_v2t": mAP_v2t})
        # 检查并保存最好的模型
        # if mAP_v2t > best_mAP:
        #     best_mAP = mAP_v2t
        #     save_model(net, "best_model_loss1_map1.pth")
        #     wandb.log({"best_mAP": best_mAP})
