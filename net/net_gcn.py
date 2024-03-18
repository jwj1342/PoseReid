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
    def __init__(self, num_classes, in_channels=3, hidden_channels=128, lstm_hidden_size=512, lstm_num_layers=1):
        super(PoseFeatureNet, self).__init__()
        # Separate GCN and LSTM for each modality
        self.gcn = GCNBlock(in_channels, hidden_channels)

        self.bn = nn.BatchNorm1d(12)

        self.lstm = nn.LSTM(hidden_channels * 17, lstm_hidden_size, lstm_num_layers, batch_first=True,
                            bidirectional=True)

        # Modality interaction
        self.interaction_layer = nn.Linear(lstm_hidden_size * 4, lstm_hidden_size * 2)  # Assuming bidirectional LSTM

        # Final classification layer
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

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

    def GCN(self, pose):
        batch_size, seq_length, num_nodes, _ = pose.size()
        gcn_outputs = []
        for time_step in range(seq_length):
            gcn_input = pose[:, time_step, :, :].reshape(-1, 3)
            gcn_output = self.gcn(gcn_input, self.connections)
            gcn_outputs.append(gcn_output.view(batch_size, num_nodes * self.gcn.conv2.out_channels))
        return torch.stack(gcn_outputs, dim=1)

    def forward(self, pose1, pose2, modal=0):
        if modal == 0:
            gcn_outputs_rgb = self.GCN(pose1)
            gcn_outputs_ir = self.GCN(pose2)
            gcn_outputs = torch.cat((gcn_outputs_rgb, gcn_outputs_ir), 0)
        else:
            gcn_outputs = self.GCN(pose1)

        gcn_outputs = self.bn(gcn_outputs)

        # LSTM
        lstm_output, _ = self.lstm(gcn_outputs)

        # Use the last time step
        lstm_output = lstm_output[:, -1, :]

        # Final classification
        feature_cls = self.classifier(lstm_output)

        return lstm_output, feature_cls


# Example usage
if __name__ == "__main__":
    pose_feature_net = PoseFeatureNet(500, 3, 128, 256, 1).cuda()
    random_rgb_pose = torch.randn(2, 12, 17, 3).cuda()  # Example: batch_size=2, seq_length=12, num_nodes=17, features=3
    random_ir_pose = torch.randn(2, 12, 17, 3).cuda()
    print(pose_feature_net)
    interaction_features, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose, modal=1)
    print("Interaction Features Shape:", interaction_features.shape)
    print("Feature Classification Shape:", feature_cls.shape)
