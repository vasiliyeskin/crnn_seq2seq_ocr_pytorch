import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

DISCR_FILTERS = 64
CNN_FEATURE_SIZE = 512


class CNN(nn.Module):
    """"
        Convolution neural network module
    """

    def __init__(self, input_shape):
        super(CNN, self).__init__()
        # this pipe converges image into the vector
        self.cnn_feature_size = CNN_FEATURE_SIZE

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS, kernel_size=3, stride=1, padding=1),
            ## (batch_size, 64, imgH, imgW)
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),  ## (batch_size, 64, imgH / 2, imgW / 2)

            nn.Conv2d(DISCR_FILTERS, 2 * DISCR_FILTERS, kernel_size=3, stride=1, padding=1),
            ## (64, 128, imgH/2, imgW/2)
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),  ## (128, 128, imgH /2/ 2, imgW /2 / 2)

            nn.Conv2d(2 * DISCR_FILTERS, 4 * DISCR_FILTERS, kernel_size=3, stride=1, padding=1),
            ## (128, 256, imgH/2/2, imgW/2/2)
            nn.BatchNorm2d(4 * DISCR_FILTERS),
            nn.ReLU(),

            nn.Conv2d(4 * DISCR_FILTERS, 4 * DISCR_FILTERS, kernel_size=3, stride=1, padding=1),
            ## (256, 256, imgH/2/2, imgW/2/2)
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0)),  ## (batch_size, 256, imgH/2/2/2, imgW/2/2)

            nn.Conv2d(4 * DISCR_FILTERS, 8 * DISCR_FILTERS, kernel_size=3, stride=1, padding=1),
            ## (256, 528, imgH/2/2/2, imgW/2/2)
            nn.BatchNorm2d(8 * DISCR_FILTERS),
            nn.ReLU(),

            nn.MaxPool2d((2, 1), stride=(2, 1), padding=(0, 0)),
            ##   batch_size, 512, imgH / 2 / 2 / 2, imgW / 2 / 2 / 2)

            nn.Conv2d(8 * DISCR_FILTERS, 8 * DISCR_FILTERS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * DISCR_FILTERS),
            nn.ReLU()
            ## (batch_size, 512, H, W)
        )

        conv_out_size = self._get_conv_out_size(input_shape)

        self.linear = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, CNN_FEATURE_SIZE)
        )

    def _get_conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        '''
        in initial version
        model: add(nn.AddConstant(-128.0))
        model: add(nn.MulConstant(1.0 / 128))
        '''
        x = x + (-128.0)  ##??
        x = x / 128.0  ##??

        '''
        in initial version
        model:add(nn.Transpose({2, 3}, {3,4})) -- (batch_size, H, W, 512)
        model:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, 512) 
        '''

        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.linear(conv_out)
