import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
    def __init__(self, num_classes, in_channels=3, hidden_channels=128, lstm_hidden_size=256, lstm_num_layers=1):
        super(PoseFeatureNet, self).__init__()
        self.gcn = GCNBlock(in_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels * 17, lstm_hidden_size, lstm_num_layers,
                            batch_first=True, bidirectional=True)  # Assuming 17 nodes per skeleton
        self.fc_pool = nn.Linear(lstm_hidden_size*2, lstm_hidden_size*2)
        self.fc_cls = nn.Linear(lstm_hidden_size*2, num_classes)
        self.connections = torch.tensor([
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6],
            [5, 7], [6, 8],
            [7, 9], [8, 10],
            [1, 2],
            [0, 1], [0, 2],
            [1, 3], [2, 4],
            [3, 5], [4, 6]
        ], dtype=torch.long).t().contiguous().cuda()

    def forward(self, pose1, pose2):
        batch_size, seq_length, num_nodes, _ = pose1.size()
        edge_index = self.connections

        # Placeholder for GCN output for each time step
        gcn_outputs = torch.empty((batch_size, seq_length, num_nodes * self.gcn.conv2.out_channels),
                                  device=pose1.device)

        for time_step in range(seq_length):
            # Process each time step for all batches together
            gcn_input = torch.cat((pose1[:, time_step, :, :], pose2[:, time_step, :, :]), dim=0)
            gcn_input = gcn_input.view(-1, 3)  # Flatten to fit GCN

            # Apply GCN
            gcn_output = self.gcn(gcn_input, edge_index)

            # Reshape and separate the concatenated batches
            gcn_output = gcn_output.view(2, batch_size, num_nodes * self.gcn.conv2.out_channels)
            gcn_output = torch.mean(gcn_output, dim=0)  # Average features from pose1 and pose2

            # Store GCN output
            gcn_outputs[:, time_step, :] = gcn_output

        # LSTM for temporal modeling
        lstm_output, _ = self.lstm(gcn_outputs)
        lstm_output = lstm_output[:, -1, :]  # Use the last time step

        # Pooling and classification
        # feature_pool = F.relu(self.fc_pool(lstm_output))
        feature_pool = F.relu(lstm_output)
        feature_cls = self.fc_cls(feature_pool)

        return feature_pool, feature_cls


# Example usage
if __name__ == "__main__":
    pose_feature_net = PoseFeatureNet(500, 3, 128, 256, 1).cuda()
    random_rgb_pose = torch.randn(2, 12, 17, 3).cuda()  # Example: batch_size=2, seq_length=12, num_nodes=17, features=3
    random_ir_pose = torch.randn(2, 12, 17, 3).cuda()
    print(pose_feature_net)
    feature_pool, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose)
    print("Feature Pool Shape:", feature_pool.shape)
    print("Feature Classification Shape:", feature_cls.shape)
