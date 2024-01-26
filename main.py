import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from imloader import CustomImageDataset

def createNet():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(200*200, 250)
            self.encoder = nn.Linear(250, 50)
            self.lat = nn.Linear(50, 250)
            self.decoder = nn.Linear(250, 200*200)
            
        #forward pass
        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.encoder(x))
            x = F.relu(self.lat(x))
            y = torch.sigmoid(self.decoder(x))
            return y
        

    net = Net()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    return net, loss_function, optimizer


net, loss_function, optimizer = createNet()

