# Targets:
# Set up the environment and build a proper CNN skeleton with Conv + Pooling + FC.
# Make the model lighter by reducing unnecessary parameters (aim <100k).
# Ensure training runs smoothly, no implementation errors.
# Achieve at least 98.5% test accuracy to establish a baseline.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Optimizer configuration for CNN_Model_2
def get_optimizer():
    return optim.SGD  # Return the actual optimizer class

def get_optimizer_params():
    return {'lr': 0.01, "momentum": 0.9}

class CNN_Model_2(nn.Module):
    def __init__(self):
        super(CNN_Model_2, self).__init__()                         # Input size: 1x28x28

        # Input Block
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False)      # 16x26x26
        self.bn1 = nn.BatchNorm2d(16)                                # BatchNorm after conv1
        self.dropout1 = nn.Dropout(0.01)                            # Dropout layer with 1% dropout rate

        # Conv Block 1
        self.conv2 = nn.Conv2d(16, 16, 3, padding=0, bias=False)    # 16x24x24
        self.bn2 = nn.BatchNorm2d(16)                                # BatchNorm after conv2
        self.dropout2 = nn.Dropout(0.05)                            # Dropout layer with 5% dropout rate
        self.conv3 = nn.Conv2d(16, 32, 3, padding=0, bias=False)    # 32x22x22
        self.bn3 = nn.BatchNorm2d(32)                               # BatchNorm after conv3

        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)                             # 32x11x11
        self.conv4 = nn.Conv2d(32, 16, 1, padding=0, bias=False)    # 16x9x9
        self.bn4 = nn.BatchNorm2d(16)                               # BatchNorm after conv4
        self.dropout3 = nn.Dropout(0.08)                            # Dropout layer with 8% dropout rate
        
        # Conv Block 2
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0, bias=False)    # 16x7x7
        self.bn5 = nn.BatchNorm2d(16)                               # BatchNorm after conv5
        self.dropout4 = nn.Dropout(0.05)                            # Dropout layer with 5% dropout rate
        self.conv6 = nn.Conv2d(16, 32, 3, padding=0, bias=False)    # 32x5x5
        self.bn6 = nn.BatchNorm2d(32)                               # BatchNorm after conv6
        self.dropout5 = nn.Dropout(0.15)                            # Dropout layer with 15% dropout rate

        # Output Block
        self.conv7 = nn.Conv2d(32, 40, 3, padding=0, bias=False)    # 40x3x3
        self.bn7 = nn.BatchNorm2d(40)                               # BatchNorm after conv7
        self.conv8 = nn.Conv2d(40, 10, 1, padding=0, bias=False)    # 10x3x3
        self.bn8 = nn.BatchNorm2d(10)                               # BatchNorm after conv8
        self.gap = nn.AdaptiveAvgPool2d((1, 1))                     # Global Average Pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # x = self.dropout3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        # x = self.dropout5(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.gap(x) 

        x = x.view(-1, 10*1*1)
        return F.log_softmax(x, dim=1)