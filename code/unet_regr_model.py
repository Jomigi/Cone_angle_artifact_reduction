import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from msd_pytorch.msd_model import (MSDModel)


# This code is copied and adapted from:
# https://github.com/milesial/Pytorch-UNet

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth):
        super(UNet, self).__init__()  #super so it can inherit from nn.Module
        self.depth=depth
        #in
        self.inc = inconv(n_channels, 64)

        #down
        down_blocks=nn.ModuleList()
        for layer in range(self.depth-1):
            down_block = down(2**(6+layer), 2**(6+layer+1))
            down_blocks.append(down_block)         
        last_layer=self.depth-1
        last_down_block=down(2**(6+last_layer), 2**(6+last_layer))
        down_blocks.append(last_down_block)
        self.layers_down= down_blocks        
        #up
        up_blocks=nn.ModuleList()
        for layer in range(self.depth-1):
            up_block=up(2**(6 + self.depth -layer), 2**(6+ self.depth-2 -layer))
            up_blocks.append(up_block)
        last_up_block=up(2**(7), 2**(6))
        up_blocks.append(last_up_block)

        self.layers_up= up_blocks

        #out
        self. outc = outconv(64, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        Hp, Wp = ((-H % 16), (-W % 16))
        padding = (Wp // 2, Wp - Wp // 2, Hp // 2, Hp - Hp // 2)
        reflect = nn.ReflectionPad2d(padding)
        x = reflect(x)
        #in
        outputs=[]
        x = self.inc(x)
        outputs.append(x)
        #down
        for layer in self.layers_down:
            x_prev=x    
            x=layer(x)
            outputs.append(x)
        #up
        reversed_outputs = outputs[::-1]
        for layer_ind, layer in enumerate(self.layers_up):
            x= layer(x, reversed_outputs[layer_ind+1])
        #out
        x = self.outc(x)       

        H2 = H + padding[2] + padding[3]
        W2 = W + padding[0] + padding[1]
        return x[:, :, padding[2]:H2-padding[3], padding[0]:W2-padding[1]]

    def clear_buffers(self):
        pass

class UNetRegressionModel(MSDModel):
    def __init__(self,network_path, c_in, c_out, depth, width, loss_function, lr, opt, dilation, reflect, conv3d):
        # Initialize msd network.
        super().__init__(c_in, c_out, 1, 1, dilation)

        loss_functions = {'L1': nn.L1Loss(),'L2': nn.MSELoss()}
    
        self.loss_function = loss_function
        self.criterion = loss_functions[loss_function]
        assert(self.criterion is not None)
        # Make Unet
        self.msd = UNet(c_in, 1, depth)

        # Initialize network
        self.net = nn.Sequential(self.scale_in, self.msd, self.scale_out)
        self.net.cuda()
    
        print(self.net, file=open(network_path +"model_summary.txt", "w"))
        model_parameters = filter(lambda p: p.requires_grad, self.msd.parameters())
        parameters = sum([np.prod(p.size()) for p in model_parameters])
        print("- parameters: %f" % parameters)

        print('Number of trainable parameters is {}'.format(parameters), file=open(network_path +"model_summary.txt", "a"))

        # Train all parameters apart from self.scale_in.
        self.lr=lr
        if opt == 'RMSprop':
            self.optimizer = optim.RMSprop(self.msd.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        elif opt == 'Adam':
            self.optimizer = optim.Adam(self.msd.parameters())

    def set_normalization(self, dataloader):
        """Normalize input data.

        This function goes through all the training data to compute
        the mean and std of the training data. It modifies the
        network so that all future invocations of the network first
        normalize input data. The normalization parameters are saved.

        :param dataloader: The dataloader associated to the training data.
        :returns:
        :rtype:

        """
        mean = 0
        square = 0
        for (data_in, _) in dataloader:
            mean += data_in.mean()
            square += data_in.pow(2).mean()

        mean /= len(dataloader)
        square /= len(dataloader)
        std = np.sqrt(square - mean ** 2)

        # The input data should be roughly normally distributed after
        # passing through net_fixed.
        self.scale_in.bias.data.fill_(- mean / std)
        self.scale_in.weight.data.fill_(1 / std)

    def set_target(self, data):
                
        # The class labels must reside on the GPU
        target = data.cuda()
        self.target = Variable(target)
