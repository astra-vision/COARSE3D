import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


# ProjectionV1
class ProjectionV1(nn.Module):
    '''
    Exploring Cross-Image Pixel Contrast for Semantic Segmentation
    '''

    def __init__(self, base_channels, proj_dim):
        super(ProjectionV1, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, proj_dim, kernel_size=1)
        )

    def forward(self, x):
        # return F.normalize(self.proj(x), p=2, dim=1)
        return self.proj(x)


# ProjectionV2
class ProjectionV2(nn.Module):
    '''
    Rethinking Semantic Segmentation: A Prototype View
    '''

    def __init__(self, base_channels, proj_dim):
        super(ProjectionV2, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, proj_dim, 1))

    def forward(self, x):
        return self.proj(x)


# ProjectionV3
class ProjectionV3(nn.Module):
    '''change from v2'''

    def __init__(self, base_channels, proj_dim):
        super(ProjectionV3, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, proj_dim, 1))

    def forward(self, x):
        return self.proj(x)


# ProjectionV4
class ProjectionV4(nn.Module):
    '''
    Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data
    '''

    def __init__(self, base_channels, proj_dim):
        super(ProjectionV4, self).__init__()
        # self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, proj_dim, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(proj_dim)
        # self.relu = nn.LeakyReLU()  # (inplace=self.inplace)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        x = torch.norm(x, p=2)
        return x
