import torch
from torch import nn
import torch.nn.functional as F

class PoseFeatureNet(nn.Module):
    def __init__(self, class_num, input_dim=6, seq_len=12, num_joints=19, lstm_hidden=256, fc_hidden=128,
                 dropout_rate=0.5):
        super(PoseFeatureNet, self).__init__()
        self.class_num = class_num
        self.dropout_rate = dropout_rate

        # Adjust input_dim to account for concatenated features from RGB and IR poses
        concat_input_dim = input_dim * 2  # Each pose feature is concatenated

        # FC layer to transform concatenated joint feature vector
        self.fc_pre_lstm = nn.Linear(concat_input_dim, fc_hidden)

        # Add Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Adjusted embed_size for LSTM input, considering the concatenated features
        self.embed_size = fc_hidden * num_joints  # New embedding size after FC layer

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.fc_after_lstm = nn.Linear(2 * lstm_hidden, class_num)

    def forward_feature_extractor(self, pose):
        batch_size, seq_len, num_joints, _ = pose.shape
        # pose: [batch_size, seq_len, num_joints, concat_input_dim]

        # Flatten to pass through the FC layer
        pose = pose.view(-1, pose.size(-1))  # Flatten to [batch_size * seq_len * num_joints, concat_input_dim]
        pose = self.fc_pre_lstm(pose)  # Pass through FC
        pose = F.relu(pose)  # Activation function
        pose = self.dropout(pose)  # Apply dropout
        pose = pose.view(batch_size, seq_len, -1)  # Reshape back

        pose, _ = self.lstm(pose)

        return pose[:, -1, :]  # Return only the last hidden state

    def forward(self, rgb_pose, ir_pose):
        # Concatenate RGB and IR poses along the feature dimension
        concat_pose = torch.cat((rgb_pose, ir_pose),
                                dim=0)  # Shape: [batch_size, seq_len, num_joints, concat_input_dim]

        feature = self.forward_feature_extractor(concat_pose)

        feature_cls = self.fc_after_lstm(feature)

        return feature, feature_cls

if __name__ == '__main__':
    """
    用于测试PoseFeatureNet的功能
    """
    pose_feature_net = PoseFeatureNet(500)
    random_rgb_pose = torch.randn(4, 12, 19, 6)
    random_ir_pose = torch.randn(4, 12, 19, 6)

    feature, feature_cls = pose_feature_net(random_rgb_pose, random_ir_pose)
    print(pose_feature_net)
    # with SummaryWriter(log_dir='./visual', comment='lstm') as writer:
    #     writer.add_graph(pose_feature_net, [random_rgb_pose, random_ir_pose])

    print(feature.shape, feature_cls.shape)