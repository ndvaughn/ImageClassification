
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (5, 5), stride=(2, 2), padding=2)
        self.conv2 = nn.Conv2d(16, 64, (5, 5), stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(64, 128, (5,5), stride=(2,2), padding=2)
        self.conv4 = nn.Conv2d(128, 32, (1, 1), stride=(1, 1))
        self.drop_layer = nn.Dropout()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 5)
        #

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(512))
        nn.init.normal_(self.fc2.weight, 0.0, 1 / sqrt(256))
        nn.init.normal_(self.fc3.weight, 0.0, 1 / sqrt(128))
        nn.init.normal_(self.fc4.weight, 0.0, 1 / sqrt(64))
        nn.init.normal_(self.fc5.weight, 0.0, 1 / sqrt(32))

        nn.init.constant(self.fc1.bias, 0.0)
        nn.init.constant(self.fc2.bias, 0.0)
        nn.init.constant(self.fc3.bias, 0.0)
        nn.init.constant(self.fc4.bias, 0.0)
        nn.init.constant(self.fc5.bias, 0.0)

        #

    def forward(self, x):
        N, C, H, W = x.shape

        z = torch.zeros([N, 5])

        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = self.drop_layer(z)
        z = F.relu(self.conv3(z))
        z = self.drop_layer(z)
        z = F.relu(self.conv4(z))
        z = z.view(N, 512)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.drop_layer(z)
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z= self.fc5(z)

        #

        return z
