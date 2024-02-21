import torch
from torch import nn


class PoseFeatureNet(nn.Module):
    def __init__(self, class_num):
        super(PoseFeatureNet, self).__init__()
        self.class_num = class_num

        self.mlp = nn.Sequential(
            nn.Linear(12 * 19 * 6, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, class_num, bias=False)
        )

        self.prediction_layer = nn.Sigmoid()

    def forward(self, rgb_pose, ir_pose):
        rgb_feature = self.mlp(rgb_pose.view(rgb_pose.size(0), -1))
        ir_feature = self.mlp(ir_pose.view(ir_pose.size(0), -1))
        rgb_prediction = self.prediction_layer(rgb_feature)
        ir_prediction = self.prediction_layer(ir_feature)
        return rgb_feature, rgb_prediction, ir_feature, ir_prediction


if __name__ == '__main__':
    # 已废弃
    pose_feature_net = PoseFeatureNet(500)
    random_rgb_pose = torch.randn(4, 12, 19, 6)
    random_ir_pose = torch.randn(4, 12, 19, 6)

    rgb_feature, rgb_prediction, ir_feature, ir_prediction = pose_feature_net(random_rgb_pose, random_ir_pose)

    print(rgb_feature, rgb_prediction, ir_feature, ir_prediction)
