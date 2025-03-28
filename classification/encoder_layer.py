# USAGE IN PAPER DESCRIPTION:
# Purpose: The encoder extracts features from the time series (uses 1D-ResNet because 1D conv layers are best for time-series type data e.g EEG classification)
# & according to the paper, using 1DResNet in the masked pre-training framework increases classificatoin accuracy.

import torch
import torch.nn.functional as F


class Conv1D(torch.nn.Module):
    """
    Main use case is used to extract features (point-wise temporal) from the time-series data like EEG.
    1D Convolutional block that implements: Conv1d, Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Conv1D, self).__init__()
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias = False)       # applies 1D conv on the input
        self.bn = torch.nn.BatchNorm1d(out_channels)                                                             # normalizes for training
        self.relu = torch.nn.ReLU(inplace = True)                                                                # applies relu activation
        

    def forward(self, x):                                                                                        # put its together to put conv, batch norm then relu
        out = self.conv(x)
        out = self.bn(x)
        return self.relu(out)
    
    

class Residual1D(torch.nn.Module):
    """
    Implements the 'residual learning' from ResNEt, it helps the network to not overfit and get stuck while training.
    Contains:
        2 conv layers and a skip connection whihc helps stablalize the deep networks.
    """

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Residual1D, self).__init__()
        
        self.conv1 = Conv1D(in_channels, out_channels, kernel_size, stride, padding)                            # 1st conv block; no downsampling
        self.conv2 = Conv1D(out_channels, in_channels, kernel_size, stride = 1, padding = padding)              # 2nd conv block; maintinas the dimension
        self.downsample = None
    
        if in_channels != out_channels or stride != 1:                                                          # identifies if input and output dimensions match, if not then adjust 
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False), 
                torch.nn.BatchNorm1d(out_channels))
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return F.relu(out)
    


class ResNetEncoder(torch.nn.Module):
    """
    The encoder extracts features by using an initial convolution and pooling method.
    Then, it stacks the residual blocks to learn. As the last step, it applies global avg pooling to 
    create a fixed-length feature vector.   
    """
    def __init__(self, in_channels = 1, base_channels = 64, block_counts = [2, 2, 2]):
        super(ResNetEncoder, self).__init__()
        
        # initial convolution and max pooling 
        self.inital_conv == torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, base_channels, kernel_size = 7, stride= 2, padding = 3, bias = False), 
            torch.nn.BatchNorm1d(base_channels),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool(kernel_size = 3, stride = 2, padding =1)
            )
        self.layer1 = self._make_layer(base_channels, base_channels, block_counts[0], stride = 1)                       # step 1: residual blocks - might start with downsampling
        self.layer2 = self._make_layer(base_channels, base_channels * 2, block_counts[1], stride = 2)                   # step 2: residual blocks - downsampling to 2x the channels
        self.layer3 = self._make_layer(base_channels * 2 , base_channels * 4 , block_counts[2], stride = 2)             # step 3: residual blocks - downsampling to 4x the channels
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool1d(1)                                                         # helps compress the time dimension to be 1 vector / series

    # helper function to build residual blocks
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Residual1D(in_channels, out_channels, stride = stride))                              # 1st block: might start with downsampling
        for _ in range(1, blocks):                                                                         # rest of the block keep the length and # of channels
            layers.append(Residual1D(out_channels, out_channels))
            return torch.nn.Sequential(* layers)
    

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        pooled = self.global_avg_pooling(x)
        flattened = pooled.view(pooled.size(0), -1)
        return flattened

    
        
        