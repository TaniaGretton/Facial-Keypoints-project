## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (68, 222, 222)
        # after one pool layer, this becomes (68, 111, 111)
        self.conv1 = nn.Conv2d(1, 68, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch             normalization) to avoid overfitting
        
        #maxpooling layer with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2,2)
        
        # second conv layer: 68 inputs, 136 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = 109
        # the output tensor will have dimensions: (136, 109, 109)
        # after another pool layer this becomes (136, 54, 54);
        self.conv2 = nn.Conv2d(68,136,3)
        
        # third conv layer: 136 inputs, 272 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (272,52,52)
        # after the pool layer this becomes (272,26,26)
        self.conv3 = nn.Conv2d(136,272,3)
        
        # fourth conv layer: 272 inputs, 544 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (544,24,24)
        # after the pool layer this becomes (544,12,12)
        self.conv4 = nn.Conv2d(272,544,3)
        
        # fifth conv layer: 544 inputs, 1088 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output tensor will have dimensions: (1088,10,10)
        # after the pool layer this becomes (1088,5,5)
        self.conv5 = nn.Conv2d(544,1088,3)
       
        # 1088 outputs * the 5*5 map size
        # class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(1088*5*5, 1000)  # 1088*5*5 = 27200
        
        # class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(1000,1000) 
        
        # dropout with p=0.4
        # class torch.nn.Dropout(p, inplace=False)
        self.dropout = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels (for the 136 keypoint x,y coord.)
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
