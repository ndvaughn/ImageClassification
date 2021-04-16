'''
EECS 445 - Introduction to Machine Learning
Winter 2020 - Project 2
CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.cnn import CNN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define each layer
        self.conv1 = nn.Conv2d(3, 16, (5,5), stride=(2,2), padding=2)
        self.conv2 = nn.Conv2d(16, 64, (5,5), stride=(2,2), padding=2)
        self.conv3 = nn.Conv2d(64, 32, (5,5), stride=(2,2), padding=2)
        self.fc1 = nn.Linear(512,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,5)
        #

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        # TODO: initialize the parameters for [self.fc1, self.fc2, self.fc3]

            nn.init.normal_(self.fc1.weight, 0.0, 1/sqrt(512))
            nn.init.normal_(self.fc2.weight, 0.0, 1/sqrt(64))
            nn.init.normal_(self.fc3.weight, 0.0, 1/sqrt(32))

            nn.init.constant(self.fc1.bias, 0.0)
            nn.init.constant(self.fc2.bias, 0.0)
            nn.init.constant(self.fc3.bias, 0.0)

        #

    def forward(self, x):
        N, C, H, W = x.shape

        z = torch.zeros([N, 5])

        # TODO: forward pass
        temp1 = F.relu(self.conv1(x))
        temp2 = F.relu(self.conv2(temp1))
        temp3 = F.relu(self.conv3(temp2))
        temp3 = temp3.view(N, 512)
        temp4 = F.relu(self.fc1(temp3))
        temp5 = F.relu(self.fc2(temp4))
        z = self.fc3(temp5)
        #

        return z
