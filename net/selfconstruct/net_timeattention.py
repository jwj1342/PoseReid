import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def calculate_edge_features(pose, connections):
    vectors = pose[:, connections[1], :2] - pose[:, connections[0], :2]  # Get vectors between joints
    lengths = vectors.norm(dim=-1).unsqueeze(-1)  # Calculate lengths
    angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0]).unsqueeze(-1)  # Calculate angles
    edge_features = torch.cat([lengths, angles], dim=-1)  # Concatenate length and angle features
    return edge_features


class GATBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8, dropout=0.6):
        super(GATBlock, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.conv2.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TemporalAttentionModule(nn.Module):
    """Temporal attention module for LSTM outputs."""
    def __init__(self, hidden_size):
        super(TemporalAttentionModule, self).__init__()
        self.hidden_size = hidden_size
        self.attention_fc = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: shape [batch_size, seq_length, hidden_size]
        attention_weights = self.attention_fc(lstm_outputs)  # Shape: [batch_size, seq_length, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # Softmax over seq_length
        attended_sequence = lstm_outputs * attention_weights  # Element-wise multiplication
        attended_sequence = torch.sum(attended_sequence, dim=1)  # Sum over seq_length
        return attended_sequence
class PoseFeatureNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=128, lstm_hidden_size=512, lstm_num_layers=1,
                 gat_heads=8, dropout=0.6):
        super(PoseFeatureNet, self).__init__()
        self.gat = GATBlock(in_channels, hidden_channels, heads=gat_heads, dropout=dropout)
        self.fc_edge_features = nn.Linear(2 * 19, hidden_channels)  # Full connection layer for edge features
        self.bn = nn.BatchNorm1d(12)

        self.lstm = nn.LSTM(hidden_channels * 19, lstm_hidden_size, lstm_num_layers, batch_first=True,
                            bidirectional=True)
        # 添加时间注意力模块
        self.temporal_attention = TemporalAttentionModule(lstm_hidden_size * 2)  # 双向LSTM的hidden_size是两倍

        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

        connections = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6],
            [5, 7], [6, 8],
            [7, 9], [8, 10],
            [1, 2],
            [0, 1], [0, 2],
            [1, 3], [2, 4],
            [3, 5], [4, 6]
        ]

        # 添加每个连接的反向连接
        connections += [[b, a] for a, b in connections]
        self.connections = torch.tensor(connections, dtype=torch.long).t().contiguous().cuda()

    def GAT(self, pose):
        batch_size, seq_length, num_nodes, _ = pose.size()
        gat_outputs = []
        for time_step in range(seq_length):
            gat_input = pose[:, time_step, :, :].reshape(-1, 3)
            gat_output = self.gat(gat_input, self.connections)
            gat_outputs.append(gat_output.view(batch_size, num_nodes * self.gat.conv2.out_channels))
        return torch.stack(gat_outputs, dim=1)

    def forward(self, pose1, pose2, modal=0):
        batch_size, seq_length, num_nodes, _ = pose1.size()
        if modal == 0:
            gat_outputs_rgb = self.GAT(pose1)
            gat_outputs_ir = self.GAT(pose2)
            edge_features_rgb = calculate_edge_features(pose1.view(-1, 17, 3), self.connections)
            edge_features_ir = calculate_edge_features(pose2.view(-1, 17, 3), self.connections)

            processed_edge_features_rgb = self.fc_edge_features(edge_features_rgb.view(-1, 2 * 19)).view(batch_size,
                                                                                                         seq_length, -1)
            processed_edge_features_ir = self.fc_edge_features(edge_features_ir.view(-1, 2 * 19)).view(batch_size,
                                                                                                       seq_length, -1)
            combined_features_rgb = torch.cat([gat_outputs_rgb, processed_edge_features_rgb], dim=-1)
            combined_features_ir = torch.cat([gat_outputs_ir, processed_edge_features_ir], dim=-1)

            combined_features = torch.cat([combined_features_rgb, combined_features_ir], dim=0)
        else:
            gat_outputs = self.GAT(pose1)
            edge_features = calculate_edge_features(pose1.view(-1, 17, 3), self.connections)
            processed_edge_features = self.fc_edge_features(edge_features.view(-1, 2 * 19)).view(batch_size, seq_length,
                                                                                                 -1)
            combined_features = torch.cat([gat_outputs, processed_edge_features], dim=-1)

        combined_features = self.bn(combined_features)
        lstm_output, _ = self.lstm(combined_features)
        # 应用时间注意力模块
        attended_lstm_output = self.temporal_attention(lstm_output)

        feature_cls = self.classifier(attended_lstm_output)

        return attended_lstm_output, feature_cls


if __name__ == "__main__":
    pose_feature_net = PoseFeatureNet(500, 3, 128, 512, 1, gat_heads=8, dropout=0.6).cuda()
    random_rgb_pose = torch.randn(2, 12, 17, 3).cuda()
    random_ir_pose = torch.randn(2, 12, 17, 3).cuda()
    interaction_features, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose, modal=0)
    print(pose_feature_net)
    print("Interaction Features Shape:", interaction_features.shape)
    print("Feature Classification Shape:", feature_cls.shape)
