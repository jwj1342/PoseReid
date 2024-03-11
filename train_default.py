import time
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_manager import VCM_Pose
from dataset_preparation import PoseDataset_train, PoseDataset_test
from net.net_attention import PoseFeatureNet as net
from util import IdentitySampler, GenIdx, load_model, save_model, test_general

pose = VCM_Pose()

rgb_pos, ir_pos = GenIdx(pose.rgb_label, pose.ir_label)

num_pos = 2
sampler = IdentitySampler(pose.ir_label, pose.rgb_label, rgb_pos, ir_pos, num_pos, 64)
index1 = sampler.index1
index2 = sampler.index2

pose_dataset = PoseDataset_train(pose.train_ir, pose.train_rgb, seq_len=12, sample='video_train', transform=None,
                                 index1=index1, index2=index2)

dataloader = DataLoader(pose_dataset, batch_size=64*num_pos, num_workers=12, drop_last=True, sampler=sampler)

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
    batch_size=32, shuffle=False, num_workers=12)

galleryloader = DataLoader(
    PoseDataset_test(pose.gallery, seq_len=12, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=12)
# ----------------visible to infrared----------------
queryloader_1 = DataLoader(
    PoseDataset_test(pose.query_1, seq_len=12, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=12)

galleryloader_1 = DataLoader(
    PoseDataset_test(pose.gallery_1, seq_len=12, sample='video_test'),
    batch_size=32, shuffle=False, num_workers=12)

if __name__ == '__main__':

    config = {
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "architecture": "LSTM&FC",
        "dataset": "VCM-POSE",
        "epochs": 91,
        "momentum":0.9,
        "name":'test'
    }
    optimizer = optim.SGD(
        net.parameters(), 
        lr=config["learning_rate"], 
        momentum=config["learning_rate"], 
        weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    best_mAP = 0
    load_model(net, "starter_model.pth")
    # 创建SummaryWriter
    writer = SummaryWriter(config["name"])

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        start_time = time.time()  # 开始时间
        net.train()
        for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(dataloader):
            input_rgb = Variable(imgs_rgb.float().to('cuda'))
            input_ir = Variable(imgs_ir.float().to('cuda'))

            label = pids_rgb

            labels = Variable(label.cuda())

            concat_feature, feature_cls = net(input_rgb, input_ir)

            loss = criterion_CE(feature_cls, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar('Metrics/loss', avg_loss, epoch)
        print(f"Epoch:{epoch}  Loss:{avg_loss} Time: {end_time - start_time} s")

        # 下面书写保存模型的代码（要求在当两个模态的map任意一个达到新高的时候进行保存，保存时候使用相关信息命名）

        if epoch % 5 == 0 and epoch != 0:
            cmc_t2v, mAP_t2v = test_general(galleryloader, queryloader, net, ngall, nquery)
            writer.add_scalar('Metrics/mAP_t2v', mAP_t2v, epoch)
            writer.add_scalar('Metrics/t2v-Rank-1', cmc_t2v[0], epoch)
            writer.add_scalar('Metrics/t2v-Rank-20', cmc_t2v[4], epoch)

            cmc_v2t, mAP_v2t = test_general(galleryloader_1, queryloader_1, net, ngall_1, nquery_1)
            writer.add_scalar('Metrics/mAP_v2t', mAP_v2t, epoch)
            writer.add_scalar('Metrics/v2t-Rank-1', cmc_v2t[0], epoch)
            writer.add_scalar('Metrics/v2t-Rank-20', cmc_v2t[4], epoch)

            if mAP_t2v + mAP_v2t > best_mAP:
                save_model(net, f"best_model_LSTM_attention_{epoch}_{best_mAP}.pth")
