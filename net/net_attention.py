import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Self attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out


class PoseFeatureNet(nn.Module):
    def __init__(self, class_num, input_dim=6, seq_len=12, num_joints=19, lstm_hidden=256, fc_hidden=128, heads=1):
        super(PoseFeatureNet, self).__init__()
        self.class_num = class_num

        self.fc_pre_lstm = nn.Linear(input_dim, fc_hidden)
        self.self_attention = SelfAttention(fc_hidden, heads)
        self.embed_size = fc_hidden * num_joints

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_after_lstm = nn.Linear(2 * lstm_hidden, class_num)
        self.classifier = nn.Linear(1024, class_num, bias=False)

    def forward_feature_extractor(self, pose):
        batch_size, seq_len, num_joints, input_dim = pose.shape

        pose = pose.view(-1, input_dim)
        pose = self.fc_pre_lstm(pose)
        pose = pose.view(batch_size * seq_len, num_joints, -1)

        # Applying self attention
        pose = self.self_attention(pose, pose, pose)
        pose = pose.view(batch_size, seq_len, -1)

        pose, _ = self.lstm(pose)

        return pose[:, -1, :]

    def forward(self, rgb_pose, ir_pose):
        rgb_feature = self.forward_feature_extractor(rgb_pose)
        ir_feature = self.forward_feature_extractor(ir_pose)

        concat_feature = torch.cat((rgb_feature, ir_feature), 1)
        feature_cls = self.classifier(concat_feature)

        return concat_feature, feature_cls


if __name__ == '__main__':
    """
    用于测试PoseFeatureNet的功能
    """
    pose_feature_net = PoseFeatureNet(500)
    random_rgb_pose = torch.randn(4, 12, 19, 6)
    random_ir_pose = torch.randn(4, 12, 19, 6)

    feature_pool, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose)
    print(pose_feature_net)
    print(feature_pool.shape, feature_cls.shape)
