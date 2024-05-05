import time
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_manager import VCM_Pose
from dataset_preparation import PoseDataset_train, PoseDataset_test
from net.st_gcn.st_gcn import Model as net
from util import IdentitySampler, GenIdx, save_model, test_general

pose = VCM_Pose()

rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)

num_pos = 2
sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, num_pos, 64)
index1 = sampler.index1
index2 = sampler.index2

pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=24, sample='video_train', transform=None,
                                 index1=index1, index2=index2)

dataloader = DataLoader(pose_dataset, batch_size=32 * num_pos, num_workers=0, drop_last=True, sampler=sampler)

criterion_CE = nn.CrossEntropyLoss()
criterion_CE.to('cuda')

net = net(in_channels=3,
          num_class=500,
          graph_args={'layout': 'coco', 'strategy': 'spatial'},
          edge_importance_weighting=True).to('cuda')

nquery_1 = pose.num_query_tracklets_1
ngall_1 = pose.num_gallery_tracklets_1
nquery = pose.num_query_tracklets
ngall = pose.num_gallery_tracklets

queryloader = DataLoader(
    PoseDataset_test(pose.query, seq_len=24, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=0)

galleryloader = DataLoader(
    PoseDataset_test(pose.gallery, seq_len=24, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=0)
# ----------------visible to infrared----------------
queryloader_1 = DataLoader(
    PoseDataset_test(pose.query_1, seq_len=24, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=0)

galleryloader_1 = DataLoader(
    PoseDataset_test(pose.gallery_1, seq_len=24, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=0)

if __name__ == '__main__':

    config = {
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "architecture": "stgcn",
        "dataset": "VCM-POSE",
        "epochs": 301,
        "momentum": 0.9,
        "name": 'stgcn'
    }
    optimizer = optim.SGD(
        net.parameters(),
        lr=config["learning_rate"],
        momentum=config["learning_rate"],
        weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=5e-4)

    best_mAP = 0.0
    # load_model(net, "best_model_GCN_moreinfo1continue_200_0.035.pth")
    # 创建SummaryWriter
    # writer = SummaryWriter("visual/" + config["name"])

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        start_time = time.time()  # 开始时间
        net.train()
        for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(dataloader):
            input1 = imgs_rgb
            input2 = imgs_ir
            label1 = pids_rgb
            label2 = pids_ir

            input_rgb = Variable(input1.float().to('cuda').permute(0, 3, 1, 2).unsqueeze(-1))
            input_ir = Variable(input2.float().to('cuda').permute(0, 3, 1, 2).unsqueeze(-1))

            labels = torch.cat((label1, label2), 0)  # 将两个模态的标签拼接起来，形成一个新的标签
            labels = Variable(labels.cuda())
            feature_rgb = net(input_rgb)
            feature_ir = net(input_ir)
            # concat_feature, feature_cls = net(input_rgb, input_ir)
            feature_cls = torch.cat((feature_rgb, feature_ir), 0)

            loss = criterion_CE(feature_cls, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()

        avg_loss = running_loss / len(dataloader)
        # writer.add_scalar('Metrics/loss', avg_loss, epoch)
        print(f"Epoch:{epoch}  Loss:{avg_loss} Time: {end_time - start_time} s")

        # 下面书写保存模型的代码（要求在当两个模态的map任意一个达到新高的时候进行保存，保存时候使用相关信息命名）

        if epoch % 10 == 0 and epoch != 0:
            net.eval()
            cmc_t2v, mAP_t2v = test_general(galleryloader, queryloader, net, ngall, nquery)
            # writer.add_scalar('Metrics/mAP_t2v', mAP_t2v, epoch)
            # writer.add_scalar('Metrics/t2v-Rank-1', cmc_t2v[0], epoch)
            # writer.add_scalar('Metrics/t2v-Rank-20', cmc_t2v[19], epoch)

            cmc_v2t, mAP_v2t = test_general(galleryloader_1, queryloader_1, net, ngall_1, nquery_1)
            # writer.add_scalar('Metrics/mAP_v2t', mAP_v2t, epoch)
            # writer.add_scalar('Metrics/v2t-Rank-1', cmc_v2t[0], epoch)
            # writer.add_scalar('Metrics/v2t-Rank-20', cmc_v2t[19], epoch)

            if mAP_t2v + mAP_v2t > best_mAP:
                best_mAP = mAP_t2v + mAP_v2t
                # save_model(net, f"best_model_GCN_moreinfo1continue_{epoch}_{round(best_mAP, 3)}.pth")
