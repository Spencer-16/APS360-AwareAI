import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class coarseNet(nn.Module):
    def __init__(self, in_channel=3, channels=(96, 256, 384, 384, 256),
                 kernel_sizes=(11, 5, 3, 3, 3), strides=(4,1,1,1,2),
                 paddings=(0,2,1,1,0), name="coarseNet"):
        super(coarseNet, self).__init__()
        self.name = name
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.Conv2d(in_channel, channels[0], kernel_sizes[0], strides[0], paddings[0]),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        ))
        self.stages.append(nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_sizes[1], strides[1], paddings[1]),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        ))
        for i in range(1, len(channels)-1):
            self.stages.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i+1], strides[i+1], paddings[i+1]),
                nn.ReLU(inplace=True)
            ))
        # (6*9*256 = 13824)
        self.fc1 = nn.Linear(13824, 6144)
        self.fc2 = nn.Linear(6144, 4800)
        # 4800 = 60*80
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        input size: [n, H, W, C] = [n, 3, 240, 320]
        3*240*320 -> 96*29*39 -> 256*14*19 -> 384*14*19 -> 384*14*19 -> 256*6*9
        """
        for stage in self.stages:
            x = stage(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 1, 60, 80)
        return x

class fineNet(nn.Module):
    def __init__(self, name="fineNet"):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 63, kernel_size = 7, stride = 2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 5, padding = 2)
        self.pool = nn.MaxPool2d(2)
        self.name = "fineNet"

    def forward(self, x, y):
        """
        input size: [n, H, W, C] = [n, 3, 240, 320]
        3*240*320 -> 63*120*160 -> 63*60*80 -> 64*60*80 -> 1*60*80
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.cat((x,y),1)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x