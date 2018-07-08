## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # after this layer we get (32, 220, 220)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        # after maxpooling, we get (32, 110, 110)
        self.pool = nn.MaxPool2d(2,2)
        # here each is in a batch of 10 
        self.conv2 = nn.Conv2d(32, 64, 3)
        # then we get (64, 108, 108)
        self.pool = nn.MaxPool2d(2,2)
        # (128, 52, 52)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # after maxpooling again, we get 128, 26, 26)
        self.conv4 = nn.Conv2d(128, 276, 1)
        # after maxpooling, we get 276, 13, 13
        # we get (20,106, 106)
        self.fc1 = nn.Linear(276*13*13, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(self.dropout1(F.relu(self.conv1(x))))
        x = self.pool(self.dropout2(F.relu(self.conv2(x))))
        x = self.pool(self.dropout3(F.relu(self.conv3(x))))
        x = self.pool(self.dropout3(F.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
