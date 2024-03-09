import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class PoseFeatureNet(nn.Module):
    def __init__(self, class_num, input_dim=6, seq_len=12, num_joints=19, lstm_hidden=256, fc_hidden=128):
        super(PoseFeatureNet, self).__init__()
        self.class_num = class_num

        # FC layer to transform each joint feature vector
        self.fc_pre_lstm = nn.Linear(input_dim, fc_hidden)

        # Adjusted embed_size for LSTM input
        self.embed_size = fc_hidden * num_joints  # New embedding size after FC layer

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.fc_after_lstm = nn.Linear(2 * lstm_hidden, class_num)

    def forward_feature_extractor(self, pose):
        batch_size, seq_len, num_joints, input_dim = pose.shape

        # Reshape to process each joint feature vector through the FC layer
        pose = pose.view(-1, input_dim)  # Flatten to [batch_size * seq_len * num_joints, input_dim]
        pose = self.fc_pre_lstm(pose)  # Pass through FC layer
        pose = pose.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, self.embed_size]

        pose, _ = self.lstm(pose)

        return pose[:, -1, :]  # Return only the last hidden state

    def forward(self, rgb_pose, ir_pose):
        rgb_feature = self.forward_feature_extractor(rgb_pose)
        ir_feature = self.forward_feature_extractor(ir_pose)

        rgb_feature_cls = self.fc_after_lstm(rgb_feature)
        ir_feature_cls = self.fc_after_lstm(ir_feature)

        feature_pool = torch.cat((rgb_feature, ir_feature), 0)
        feature_cls = torch.cat((rgb_feature_cls, ir_feature_cls), 0)

        return feature_pool, feature_cls


if __name__ == '__main__':
    """
    用于测试PoseFeatureNet的功能
    """
    pose_feature_net = PoseFeatureNet(500)
    random_rgb_pose = torch.randn(4, 12, 19, 6)
    random_ir_pose = torch.randn(4, 12, 19, 6)

    feature_pool, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose)
    print(pose_feature_net)
    # with SummaryWriter(log_dir='../visual', comment='lstm') as writer:
    #     writer.add_graph(pose_feature_net, [random_rgb_pose, random_ir_pose])

    print(feature_pool.shape, feature_cls.shape)
