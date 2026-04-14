from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mmcv.ops import CARAFEPack
from timm.models import load_checkpoint
from torch import nn
from mmcv.ops.carafe import CARAFEPack
import torch
import torch.nn as nn
import torch.nn.functional as F




__all__ = ['UNet', 'NestedUNet','BiFormer_resnet18']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # self.conv1 = DeformConv2D(in_channels, middle_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        # self.conv2 = DeformConv2D(middle_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [ 64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


# ... existing code ...
from Allmodels.biformer.biformer import BiFormer
from timm.models.layers import LayerNorm2d
import torchvision.models as models

# # BRnet
from mmcv.ops.carafe import CARAFEPack
class SimAM(nn.Module):
    """
    SIMAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    Official implementation from: https://github.com/ZjjConan/SimAM
    """

    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1
        # Calculate the variance of the feature map
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class BiFormer_resnet18(BiFormer):
    def __init__(self, num_classes=1, pretrained=None, **kwargs):
        super().__init__(**kwargs)

        del self.head
        del self.norm

        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))

        resnet = models.resnet18(pretrained=True)
        self.resnet_init = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.resnet_maxpool = resnet.maxpool

        self.resnet_layers = nn.ModuleList([
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ])

        self.resnet_adapters = nn.ModuleList()
        res_channels = [64, 128, 256, 512]
        for i in range(4):
            self.resnet_adapters.append(
                nn.Conv2d(res_channels[i], self.embed_dim[i], kernel_size=1)
            )


        self.scale_attention = nn.ModuleList()
        for i in range(4):
            self.scale_attention.append(
                SimAM(channels=self.embed_dim[i])
            )
        # VSC-PFM
        self.multiscale_fusion = nn.ModuleList()
        for i in range(3):  
            self.multiscale_fusion.append(
                nn.Sequential(
                    nn.Conv2d(self.embed_dim[i] + self.embed_dim[i + 1], self.embed_dim[i], kernel_size=1),
                    nn.BatchNorm2d(self.embed_dim[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dim[i], self.embed_dim[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.embed_dim[i]),
                    nn.ReLU(inplace=True)
                )
            )

        # USE CARAFE
        # CARAFE参数说明:
        #   channels: 输入通道数
        #   scale_factor: 上采样倍数
        #   up_kernel: 上采样核大小
        #   up_group: 分组数
        #   encoder_kernel: 编码器核大小
        self.upsample_3_to_2 = nn.Sequential(
            nn.Conv2d(self.embed_dim[3], self.embed_dim[2], kernel_size=1),
            CARAFEPack(channels=self.embed_dim[2], scale_factor=2, up_kernel=5, up_group=1, encoder_kernel=3)
        )

        self.upsample_2_to_1 = nn.Sequential(
            nn.Conv2d(2 * self.embed_dim[2], self.embed_dim[1], kernel_size=1),
            CARAFEPack(channels=self.embed_dim[1], scale_factor=2, up_kernel=5, up_group=1, encoder_kernel=3)
        )

        self.upsample_1_to_0 = nn.Sequential(
            nn.Conv2d(2 * self.embed_dim[1], self.embed_dim[0], kernel_size=1),
            CARAFEPack(channels=self.embed_dim[0], scale_factor=2, up_kernel=5, up_group=1, encoder_kernel=3)
        )

        self.final_upsample = CARAFEPack(channels=2 * self.embed_dim[0], scale_factor=2, up_kernel=5, up_group=1,
                                         encoder_kernel=3)

        self.final_conv = nn.Conv2d(2 * self.embed_dim[0], num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor):
        out = []
        x_resnet = self.resnet_init(x)
        x_resnet = self.resnet_maxpool(x_resnet)
        x_biformer = x

        # 编码器前向传播和特征融合
        fused_features = []
        for i in range(4):
            x_biformer = self.downsample_layers[i](x_biformer)
            x_biformer = self.stages[i](x_biformer)
            x_resnet = self.resnet_layers[i](x_resnet)
            x_resnet_adapted = self.resnet_adapters[i](x_resnet)
            fused = x_biformer + x_resnet_adapted

            attended_features = self.scale_attention[i](fused)

            fused_features.append(self.extra_norms[i](attended_features))

        # VSC-PFM
        enhanced_features = [None] * 4
        enhanced_features[3] = fused_features[3]

        for i in range(2, -1, -1):
            upsampled = F.interpolate(enhanced_features[i + 1], size=fused_features[i].size()[2:],
                                      mode='bilinear', align_corners=True)

            concatenated = torch.cat([fused_features[i], upsampled], dim=1)
            enhanced_features[i] = self.multiscale_fusion[i](concatenated)

        return enhanced_features

    def forward(self, x: torch.Tensor):
        features = self.forward_features(x)

        feat_3 = features[3]
        feat_2 = features[2]
        feat_1 = features[1]
        feat_0 = features[0]

        # 使用CARAFE进行上采样
        up_3_to_2 = self.upsample_3_to_2(feat_3)
        combined_2 = torch.cat([up_3_to_2, feat_2], dim=1)

        up_2_to_1 = self.upsample_2_to_1(combined_2)
        combined_1 = torch.cat([up_2_to_1, feat_1], dim=1)

        up_1_to_0 = self.upsample_1_to_0(combined_1)
        combined_0 = torch.cat([up_1_to_0, feat_0], dim=1)

        output = self.final_upsample(combined_0)
        output = self.final_conv(output)

        # 确保输出大小与输入一致
        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=True)

        return output


