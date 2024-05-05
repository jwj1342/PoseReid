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
    def __init__(self, in_channels,hidden_channels,dropout=0.6, heads=8):
        super(GATBlock, self).__init__()
        self.heads = heads  # Assuming the same number of heads for simplicity

        # First convolution increases the feature size to 64
        self.conv1 = GATConv(in_channels, 64, heads=self.heads, dropout=dropout, concat=True)

        # Second convolution increases the feature size to 128
        self.conv2 = GATConv(64 * self.heads, 128, heads=self.heads, dropout=dropout,
                             concat=True)  # Input channels need to be adjusted based on concat=True

        # Third convolution increases the feature size to 256
        self.conv3 = GATConv(128 * self.heads, 256, heads=self.heads, dropout=dropout,
                             concat=False)  # Here we assume concat=False for the final layer, adjust as necessary

    def forward(self, x, edge_index):
        # Pass through the first convolution
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        # Pass through the second convolution
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        # Pass through the third convolution
        x = self.conv3(x, edge_index)
        # Final activation can be added here if necessary, e.g., x = F.elu(x)

        return x


class PoseFeatureNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=128, lstm_hidden_size=512, lstm_num_layers=1,
                 gat_heads=8, dropout=0.6):
        super(PoseFeatureNet, self).__init__()
        self.gat = GATBlock(in_channels, hidden_channels, heads=gat_heads,dropout=0.6)
        self.fc_edge_features = nn.Linear(2 * 19, hidden_channels)  # Full connection layer for edge features
        self.bn = nn.BatchNorm1d(24)

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

    def GAT(self, pose):
        batch_size, seq_length, num_nodes, _ = pose.size()
        gat_outputs = []
        for time_step in range(seq_length):
            gat_input = pose[:, time_step, :, :].reshape(-1, 3)
            gat_output = self.gat(gat_input, self.connections)
            gat_outputs.append(gat_output.view(batch_size, num_nodes * 256))
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
        lstm_output = lstm_output[:, -1, :]
        feature_cls = self.classifier(lstm_output)

        return lstm_output, feature_cls


if __name__ == "__main__":
    pose_feature_net = PoseFeatureNet(500, 3, 128, 512, 1, gat_heads=8, dropout=0.6).cuda()
    random_rgb_pose = torch.randn(2, 24, 17, 3).cuda()
    random_ir_pose = torch.randn(2, 24, 17, 3).cuda()
    interaction_features, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose, modal=0)
    print(pose_feature_net)
    print("Interaction Features Shape:", interaction_features.shape)
    print("Feature Classification Shape:", feature_cls.shape)
