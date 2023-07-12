import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class coarseNet(nn.Module):
    def __init__(self, in_channel=3, channels=(96, 256, 384, 384, 256),
                 kernel_sizes=(11, 5, 3, 3, 3), strides=(4,1,1,1,2),
                 paddings=(0,2,1,1,0), name="coarseNet"):
        super(coarseNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0)
        # self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, padding = 2)
        # self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
        # self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 2)
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
        
        self.x, self.y = self.compute_fc_input(576, 448, channels=channels,
                                               kernel_sizes=kernel_sizes, strides=strides,
                                               paddings=paddings)
        fc_input = self.x * self.y * channels[-1]
        # (17*13*256 = 56576)
        self.fc1 = nn.Linear(fc_input, 8192)
        self.fc2 = nn.Linear(8192, 3905)
        # 3905 = 55*71
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d()

    def compute_fc_input(input_x, input_y, channels=(96, 256, 384, 384, 256), 
                         kernel_sizes=(11, 5, 3, 3, 3), strides=(4,1,1,1,2),
                         paddings=(0,2,1,1,0)):
        input_x = math.floor((input_x-kernel_sizes[0]+2*paddings[0])/strides[0])
        input_x += 1
        input_x  = math.floor(input_x/2)
        input_x = math.floor((input_x-kernel_sizes[1]+2*paddings[1])/strides[1])
        input_x += 1
        input_x  = math.floor(input_x/2)
        for i in range(2, len(kernel_sizes)):
            input_x = math.floor((input_x-kernel_sizes[i]+2*paddings[i])/strides[i])
            input_x += 1

        input_y = math.floor((input_y-kernel_sizes[0]+2*paddings[0])/strides[0])
        input_y += 1
        input_y  = math.floor(input_y/2)
        input_y = math.floor((input_y-kernel_sizes[1]+2*paddings[1])/strides[1])
        input_y += 1
        input_y  = math.floor(input_y/2)
        for i in range(2, len(kernel_sizes)):
            input_y = math.floor((input_y-kernel_sizes[i]+2*paddings[i])/strides[i])
            input_y += 1

        output_x, output_y = input_x, input_y
        return output_x, output_y
        


    def forward(self, x):
        """
        input size: [n, H, W, C] = [n, 448, 576, 3]
        448*576*3 -> 55*71*96 -> 27*35*256 -> 27*35*384 -> 27*35*384 -> 13*17*256
        """
        for stage in self.stages:
            x = stage(x)
        x.xview(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 55, 71)
        return x
        #                                         # [n, c,  H,   W ]
        #                                         # [8, 3, 228, 304]
        # x = self.conv1(x)                       # [8, 96, 55, 74]
        # x = F.relu(x)
        # x = self.pool(x)                        # [8, 96, 27, 37] -- 
        # x = self.conv2(x)                       # [8, 256, 23, 33]
        # x = F.relu(x)
        # x = self.pool(x)                        # [8, 256, 11, 16] 18X13
        # x = self.conv3(x)                       # [8, 384, 9, 14]
        # x = F.relu(x)
        # x = self.conv4(x)                       # [8, 384, 7, 12]
        # x = F.relu(x)
        # x = self.conv5(x)                       # [8, 256, 5, 10] 8X5
        # x = F.relu(x)
        # x = x.view(x.size(0), -1)               # [8, 12800]
        # x = F.relu(self.fc1(x))                 # [8, 4096]
        # x = self.dropout(x)
        # x = self.fc2(x)                         # [8, 4070]     => 55x74 = 4070
        # x = x.view(-1, 1, 55, 74)
        # return x
                
                
class fineNet(nn.Module):
    def __init__(self, name="fineNet"):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, stride = 2)
        self.conv2 = nn.Conv2d(64, 63, kernel_size = 3, padding = 1, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(size=(448, 576), mode="bilinear")
        self.name = "fineNet"

    def forward(self, x, y):
        """
        input size: [n, H, W, C] = [n, 448, 576, 3]
        448*576*3 -> 110*142*64 -> 55*71*63 (cat) -> 55*71*64 -> 55*71*1
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.cat((x,y),1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)
        return x