import time
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_manager import VCM_Pose
from dataset_preparation import PoseDataset_train, PoseDataset_test
from net.net_lstm import PoseFeatureNet as net
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
    features = np.zeros((2*len(data_loader.dataset), feature_dimension))
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

# ______________
pose = VCM_Pose()

rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)

sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, 2, 64)
index1 = sampler.index1
index2 = sampler.index2

pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=12, sample='video_train', transform=None,
                                 index1=index1, index2=index2)

dataloader = DataLoader(pose_dataset, batch_size=64, num_workers=0, drop_last=True, sampler=sampler)

criterion_CE = nn.CrossEntropyLoss()
criterion_CE.to('cuda')

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

if __name__ == '__main__':

    config = {
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "architecture": "LSTM&FC",
        "dataset": "VCM-POSE",
        "epochs": 1,
    }
    optimizer = optim.SGD(net.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    best_mAP = 0
    # load_model(net, "best_model_LSTMmore.pth")
    # 创建SummaryWriter
    writer = SummaryWriter('./visual/LSTM')

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        start_time = time.time()  # 开始时间
        net.train()
        for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(dataloader):
            input_rgb = Variable(imgs_rgb.float().to('cuda'))
            input_ir = Variable(imgs_ir.float().to('cuda'))

            label1 = pids_rgb
            label2 = pids_ir

            labels = torch.cat((label1, label2), 0)  # 将两个模态的标签拼接起来，形成一个新的标签
            labels = Variable(labels.cuda())

            rgb_feature, ir_feature = net(input_rgb, input_ir)

            loss = criterion_CE(rgb_feature, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar('Metrics/loss', avg_loss, epoch)
        print(f"Epoch:{epoch}  Loss:{avg_loss} Time: {end_time - start_time} s")

        # 下面书写保存模型的代码（要求在当两个模态的map任意一个达到新高的时候进行保存，保存时候使用相关信息命名）

        if epoch % 10 == 0:
            cmc_t2v, mAP_t2v = test_general(galleryloader, queryloader, net, ngall, nquery)
            writer.add_scalar('Metrics/mAP_t2v', mAP_t2v, epoch)
            writer.add_scalar('Metrics/t2v-Rank-1', cmc_t2v[0], epoch)
            writer.add_scalar('Metrics/t2v-Rank-20', cmc_t2v[4], epoch)

            cmc_v2t, mAP_v2t = test_general(galleryloader_1, queryloader_1, net, ngall_1, nquery_1)
            writer.add_scalar('Metrics/mAP_v2t', mAP_v2t, epoch)
            writer.add_scalar('Metrics/v2t-Rank-1', cmc_v2t[0], epoch)
            writer.add_scalar('Metrics/v2t-Rank-20', cmc_v2t[4], epoch)

            # if mAP_t2v + mAP_v2t > best_mAP:
            #     save_model(net, f"best_model_LSTM_{epoch}_{best_mAP}.pth")
