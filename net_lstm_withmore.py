import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamPoseNet(nn.Module):
    def __init__(self, class_num, input_dim=6, seq_len=12, num_joints=19, lstm_hidden=256, fc_hidden=128):
        super(DualStreamPoseNet, self).__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.fc_hidden = fc_hidden

        # FC layer for spatial feature extraction
        self.fc_spatial = nn.Linear(input_dim * num_joints, fc_hidden)

        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(input_size=input_dim * num_joints, hidden_size=lstm_hidden, num_layers=1, batch_first=True,
                            bidirectional=True)

        # FC layer after LSTM
        self.fc_temporal = nn.Linear(2 * lstm_hidden, fc_hidden)

        # Final FC layer to combine features
        self.fc_final = nn.Linear(fc_hidden * (seq_len + 1), class_num)  # seq_len spatial features + 1 temporal feature

    def forward_feature_extractor(self, pose):
        batch_size, seq_len, num_joints, input_dim = pose.shape

        # Extract spatial features for each sequence
        spatial_features = pose.view(batch_size * seq_len, -1)  # Flatten each sequence
        spatial_features = F.relu(self.fc_spatial(spatial_features))  # [batch_size * seq_len, fc_hidden]
        spatial_features = spatial_features.view(batch_size, seq_len,
                                                 -1)  # Reshape back to [batch_size, seq_len, fc_hidden]

        # Extract temporal features
        temporal_input = pose.view(batch_size, seq_len, -1)  # Flatten each sequence for LSTM input
        temporal_features, _ = self.lstm(temporal_input)  # [batch_size, seq_len, 2 * lstm_hidden]
        temporal_features = F.relu(self.fc_temporal(temporal_features[:, -1, :]))  # Use only the last hidden state

        # Combine spatial and temporal features
        combined_features = torch.cat([spatial_features.view(batch_size, -1), temporal_features],
                                      dim=1)  # Flatten spatial features and concatenate

        return combined_features

    def forward(self, rgb_pose, ir_pose):
        rgb_feature = self.forward_feature_extractor(rgb_pose)
        ir_feature = self.forward_feature_extractor(ir_pose)

        # Combine features from both modalities for final classification
        rgb_output = self.fc_final(rgb_feature)
        ir_output = self.fc_final(ir_feature)

        if self.training:
            return rgb_output, ir_output
        else:
            return rgb_output


if __name__ == '__main__':
    """
    Test the DualStreamPoseNet functionality
    """
    pose_net = DualStreamPoseNet(class_num=500)
    random_rgb_pose = torch.randn(4, 12, 19, 6)
    random_ir_pose = torch.randn(4, 12, 19, 6)

    rgb_output, ir_output = pose_net(random_rgb_pose, random_ir_pose)

    print(pose_net)
    print(rgb_output.shape, ir_output.shape)
