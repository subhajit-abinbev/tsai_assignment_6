# Targets:
# Set up the environment and build a proper CNN skeleton with Conv + Pooling + FC.
# Make the model lighter by reducing unnecessary parameters (aim <100k).
# Ensure training runs smoothly, no implementation errors.
# Achieve at least 98.5% test accuracy to establish a baseline.

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model_1(nn.Module):
    def __init__(self):
        super(CNN_Model_1, self).__init__()                         # Input size: 1x28x28
        self.conv1 = nn.Conv2d(1, 8, 3, padding=0, bias=False)      # 8x26x26
        self.conv2 = nn.Conv2d(8, 16, 3, padding=0, bias=False)     # 16x24x24
        self.conv3 = nn.Conv2d(16, 32, 3, padding=0, bias=False)    # 32x22x22
        self.pool1 = nn.MaxPool2d(2, 2)                             # 32x11x11

        self.conv4 = nn.Conv2d(32, 64, 3, padding=0, bias=False)    # 64x9x9
        self.conv5 = nn.Conv2d(64, 64, 3, padding=0, bias=False)    # 64x7x7
        self.conv6 = nn.Conv2d(64, 32, 3, padding=0, bias=False)    # 32x5x5
        self.conv7 = nn.Conv2d(32, 32, 1, padding=0, bias=False)    # 32x5x5
        self.conv8 = nn.Conv2d(32, 10, 5, padding=0, bias=False)    # 10x1x1

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)

        x = x.view(-1, 10*1*1)
        return F.log_softmax(x, dim=1)