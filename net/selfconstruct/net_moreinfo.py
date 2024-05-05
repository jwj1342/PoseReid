import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def calculate_edge_features(pose, connections):
    vectors = pose[:, connections[1], :2] - pose[:, connections[0], :2]  # Get vectors between joints
    lengths = vectors.norm(dim=-1).unsqueeze(-1)  # Calculate lengths
    angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0]).unsqueeze(-1)  # Calculate angles
    edge_features = torch.cat([lengths, angles], dim=-1)  # Concatenate length and angle features
    return edge_features


class GCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class PoseFeatureNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=128, lstm_hidden_size=512, lstm_num_layers=1):
        super(PoseFeatureNet, self).__init__()
        self.gcn = GCNBlock(in_channels, hidden_channels)
        self.fc_edge_features = nn.Linear(2 * 19, hidden_channels)  # Full connection layer for edge features
        self.bn = nn.BatchNorm1d(12)

        self.lstm = nn.LSTM(hidden_channels * 19, lstm_hidden_size, lstm_num_layers, batch_first=True,
                            bidirectional=True)

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


    def GCN(self, pose):
        batch_size, seq_length, num_nodes, _ = pose.size()
        gcn_outputs = []
        for time_step in range(seq_length):
            gcn_input = pose[:, time_step, :, :].reshape(-1, 3)
            gcn_output = self.gcn(gcn_input, self.connections)
            gcn_outputs.append(gcn_output.view(batch_size, num_nodes * self.gcn.conv2.out_channels))
        return torch.stack(gcn_outputs, dim=1)

    def forward(self, pose1, pose2, modal=0):
        batch_size, seq_length, num_nodes, _ = pose1.size()
        if modal == 0:
            gcn_outputs_rgb = self.GCN(pose1)
            gcn_outputs_ir = self.GCN(pose2)
            edge_features_rgb = calculate_edge_features(pose1.view(-1, 17, 3), self.connections)
            edge_features_ir = calculate_edge_features(pose2.view(-1, 17, 3), self.connections)

            processed_edge_features_rgb = self.fc_edge_features(edge_features_rgb.view(-1, 2 * 19)).view(batch_size, seq_length, -1)
            processed_edge_features_ir = self.fc_edge_features(edge_features_ir.view(-1, 2 * 19)).view(batch_size, seq_length, -1)

            combined_features_rgb = torch.cat([gcn_outputs_rgb, processed_edge_features_rgb], dim=-1)
            combined_features_ir = torch.cat([gcn_outputs_ir, processed_edge_features_ir], dim=-1)

            combined_features = torch.cat([combined_features_rgb, combined_features_ir], dim=0)
        else:
            gcn_outputs = self.GCN(pose1)
            edge_features = calculate_edge_features(pose1.view(-1, 17, 3), self.connections)
            processed_edge_features = self.fc_edge_features(edge_features.view(-1, 2 * 19)).view(batch_size, seq_length, -1)
            combined_features = torch.cat([gcn_outputs, processed_edge_features], dim=-1)

        combined_features = self.bn(combined_features)
        lstm_output, _ = self.lstm(combined_features)
        lstm_output = lstm_output[:, -1, :]
        feature_cls = self.classifier(lstm_output)

        return lstm_output, feature_cls


if __name__ == "__main__":
    pose_feature_net = PoseFeatureNet(500, 3, 128, 512, 1).cuda()
    random_rgb_pose = torch.randn(2, 12, 17, 3).cuda()
    random_ir_pose = torch.randn(2, 12, 17, 3).cuda()
    interaction_features, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose, modal=0)
    print(pose_feature_net)
    print("Interaction Features Shape:", interaction_features.shape)
    print("Feature Classification Shape:", feature_cls.shape)
