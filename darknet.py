import torch
import torch.nn as nn

def conv3x3(inplanes, outplanes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(inplanes, outplanes, bias=False):
    """1x1 convolution without padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, downsample=False):
        super(BasicBlock, self).__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                conv3x3(inplanes, inplanes * 2, stride=2), 
                nn.BatchNorm2d(inplanes * 2),
                nn.LeakyReLU(negative_slope=1e-1, inplace=True)
            )
            inplanes = inplanes * 2

        self.conv_1 = conv1x1(inplanes, outplanes // 2) 
        self.batch_norm_1 = nn.BatchNorm2d(outplanes // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.leaky_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_2 = conv3x3(outplanes // 2, outplanes)
        self.batch_norm_2 = nn.BatchNorm2d(outplanes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.leaky_2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.leaky_1(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = self.leaky_2(out)

        return x + out

class TailBlock(nn.Module):

    def __init__(self, inplanes, outplanes, upsample=False, output=False):
        super(TailBlock, self).__init__()
        self.conv_1 = conv1x1(inplanes, outplanes // 2) 
        self.batch_norm_1 = nn.BatchNorm2d(outplanes // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.leaky_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_2 = conv3x3(outplanes // 2, outplanes)
        self.batch_norm_2 = nn.BatchNorm2d(outplanes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.leaky_2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.output = None
        if output:
            self.output = conv1x1(outplanes, 255, bias=True)

        self.upsample = None
        if upsample:
            self.upsample = nn.Sequential(
                conv1x1(outplanes // 2, outplanes // 4),
                nn.BatchNorm2d(outplanes // 4),
                nn.LeakyReLU(negative_slope=1e-1, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.leaky_1(x)

        out = self.conv_2(x)
        out = self.batch_norm_2(out)
        out = self.leaky_2(out)
        
        if self.output is not None:
            out = self.output(out) 
        
        if self.upsample is not None:
            return out, self.upsample(x)
        else:
            return out

class Darknet(nn.Module):

    def __init__(self, layers):
        super(Darknet, self).__init__()
        self.head = nn.Sequential(
            conv3x3(3, 32), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=1e-1, inplace=True),
        )

        self.layer_0 = self.__make_layer(BasicBlock, 32, 64, layers[0])
        self.layer_1 = self.__make_layer(BasicBlock, 64, 128, layers[1])
        self.layer_2 = self.__make_layer(BasicBlock, 128, 256, layers[2])

        self.layer_3 = self.__make_layer(BasicBlock, 256, 512, layers[3])
        self.layer_4 = self.__make_layer(BasicBlock, 512, 1024, layers[4])

        self.output_0 = self.__make_layer(TailBlock, 1024, 1024, layers[5])
        self.output_1 = self.__make_layer(TailBlock, 768, 512, layers[6])
        self.output_2 = self.__make_layer(TailBlock, 384, 256, layers[7], resample=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, inplanes, outplanes, blocks, resample=True):
        if block.__name__ == 'BasicBlock':
            layers = []
            layers.append(block(inplanes, outplanes, downsample=resample))
            for i in range(1, blocks):
                layers.append(block(outplanes, outplanes))
            return nn.Sequential(*layers)
        elif block.__name__ == 'TailBlock':
            layers = []
            layers.append(block(inplanes, outplanes))
            for i in range(2, blocks):
                layers.append(block(outplanes, outplanes))
            layers.append(block(outplanes, outplanes, upsample=resample, output=True))
            return nn.Sequential(*layers)
        else:
            raise NotImplementedError(block.__name__)

    def forward(self, x):
        x = self.head(x)
        x = self.layer_0(x)
        x = self.layer_1(x)

        p2 = self.layer_2(x)
        p1 = self.layer_3(p2)
        p0 = self.layer_4(p1)

        out0, p0 = self.output_0(p0)
        out1, p1 = self.output_1(torch.cat((p0, p1), dim=1))
        out2 = self.output_2(torch.cat((p1, p2), dim=1))

        return out0, out1, out2

def darknet53(filepath=None):
    """
    Constructs a Darknet-53 model for YOLOv3.
    Args:
        pretrained (bool): If True, returns a pre-trained model
    """
    model = Darknet([1, 2, 8, 8, 4, 3, 3, 3])
    if filepath is not None:
        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict)
    return model