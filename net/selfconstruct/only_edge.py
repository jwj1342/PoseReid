import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_edge_features(pose, connections):
    vectors = pose[:, connections[1], :2] - pose[:, connections[0], :2]  # Get vectors between joints
    lengths = vectors.norm(dim=-1).unsqueeze(-1)  # Calculate lengths
    angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0]).unsqueeze(-1)  # Calculate angles
    edge_features = torch.cat([lengths, angles], dim=-1)  # Concatenate length and angle features
    return edge_features


class PoseFeatureNet(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=512, lstm_num_layers=1):
        super(PoseFeatureNet, self).__init__()
        self.fc_edge_features = nn.Linear(19 * 2, 128)  # Adjust input size for edge features
        self.bn = nn.BatchNorm1d(12)
        self.lstm = nn.LSTM(128, lstm_hidden_size, lstm_num_layers, batch_first=True, bidirectional=True)
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

    def forward(self, pose1, pose2, modal=0):
        batch_size, seq_length, _, _ = pose1.size()

        edge_features1 = calculate_edge_features(pose1.view(-1, 17, 3), self.connections)
        edge_features2 = calculate_edge_features(pose2.view(-1, 17, 3), self.connections)

        processed_edge_features1 = (self.fc_edge_features(edge_features1.view(-1, 19 * 2))
                                    .view(batch_size, seq_length, -1))
        processed_edge_features2 = (self.fc_edge_features(edge_features2.view(-1, 19 * 2))
                                    .view(batch_size, seq_length, -1))

        combined_features = torch.cat([processed_edge_features1, processed_edge_features2],
                                      dim=0) if modal == 0 else processed_edge_features1
        combined_features = self.bn(combined_features)

        lstm_output, _ = self.lstm(combined_features)
        lstm_output = lstm_output[:, -1, :]
        feature_cls = self.classifier(lstm_output)

        return lstm_output, feature_cls


if __name__ == "__main__":
    pose_feature_net = PoseFeatureNet(500, 512, 1).cuda()
    random_rgb_pose = torch.randn(2, 12, 17, 3).cuda()
    random_ir_pose = torch.randn(2, 12, 17, 3).cuda()
    interaction_features, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose, modal=1)
    print("Interaction Features Shape:", interaction_features.shape)
    print("Feature Classification Shape:", feature_cls.shape)
