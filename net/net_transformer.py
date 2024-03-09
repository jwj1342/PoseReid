import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将位置编码矩阵注册为模型的参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 添加位置编码
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerFeatureNet(nn.Module):
    def __init__(self, class_num, input_dim=6, seq_len=12, num_joints=19, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048):
        super(TransformerFeatureNet, self).__init__()
        self.class_num = class_num
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_pre_transformer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_after_transformer = nn.Linear(d_model * num_joints, class_num)

    def forward_feature_extractor(self, pose):
        batch_size, seq_len, num_joints, input_dim = pose.shape
        pose = pose.view(-1, input_dim)
        pose = self.fc_pre_transformer(pose)
        pose = pose.view(batch_size, seq_len, -1)
        pose = pose.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, feature]
        pose = self.pos_encoder(pose)
        pose = self.transformer_encoder(pose)
        pose = pose.permute(1, 0, 2).contiguous().view(batch_size, -1)  # Flatten for the final FC layer
        return pose

    def forward(self, rgb_pose, ir_pose):
        rgb_feature = self.forward_feature_extractor(rgb_pose)
        ir_feature = self.forward_feature_extractor(ir_pose)

        rgb_feature = self.fc_after_transformer(rgb_feature)
        ir_feature = self.fc_after_transformer(ir_feature)

        return rgb_feature, ir_feature


if __name__ == '__main__':
    pose_feature_net = TransformerFeatureNet(500)

    random_rgb_pose = torch.randn(4, 12, 19, 6)
    random_ir_pose = torch.randn(4, 12, 19, 6)

    rgb_feature, ir_feature = pose_feature_net(random_rgb_pose, random_ir_pose)
    print(pose_feature_net)

    print(rgb_feature.shape, ir_feature.shape)
