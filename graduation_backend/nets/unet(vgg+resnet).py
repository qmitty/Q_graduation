import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16


class FuseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FuseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# 以下是multimodal的unet

class Unet(nn.Module):

    def __init__(self, num_classes=21, pretrained=False):
        super(Unet, self).__init__()
        # 模态A的编码器 (VGG16)
        self.encoder_A = VGG16(pretrained=pretrained)
        # 模态B的编码器 (ResNet50 对不起了resnet50我回头再用你，我改不出来一样的通道数好让你俩特征图融合ww)
        self.encoder_B = resnet50(pretrained=pretrained)
        # self.encoder_B = VGG16(pretrained=pretrained)
        
        # 请根据具体的模型输出自行调整这些过滤器大小
        in_filters_A = [64, 192, 384, 768]
        in_filters_B = [64, 192, 384, 768]
        # 假设在融合之后，通道数将会翻倍，因此out_filters需要调整
        out_filters = [128, 256, 512, 1024]
        
        
        # 初始化融合层（分层融合策略，偏中期融合）
        self.fuse1 = FuseConv(in_filters_A[0] + in_filters_B[0], out_filters[0])
        self.fuse2 = FuseConv(in_filters_A[1] + in_filters_B[1], out_filters[1])
        self.fuse3 = FuseConv(in_filters_A[2] + in_filters_B[2], out_filters[2])
        self.fuse4 = FuseConv(in_filters_A[3] + in_filters_B[3], out_filters[3])

        # 定义融合后的上采样层
        self.up_concat4 = unetUp(in_filters_A[3] + in_filters_B[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters_A[2] + in_filters_B[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters_A[1] + in_filters_B[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters_A[0] + in_filters_B[0], out_filters[0])
        
        # 定义最终的卷积层来映射到类别数
        self.final = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, input_A, input_B):
        # 分别对两种模态的输入进行特征提取 input_A和input_B分别是模态A和模态B的输入数据。
        feats_A = self.encoder_A(input_A)
        feats_B = self.encoder_B(input_B)
        
        # 先将feats_B的特征图尺寸都向上采样一倍，以和A尺寸保持一致
        feats_B_0_upsampled = F.interpolate(feats_B[0], size=(512, 512), mode='bilinear', align_corners=False)
        feats_B_1_upsampled = F.interpolate(feats_B[1], scale_factor=2, mode='bilinear', align_corners=False)
        feats_B_2_upsampled = F.interpolate(feats_B[2], scale_factor=2, mode='bilinear', align_corners=False)
        feats_B_3_upsampled = F.interpolate(feats_B[3], scale_factor=2, mode='bilinear', align_corners=False)
        # 跳跃连接 特征融合
        fuse_feat1 = self.fuse1(torch.cat((feats_A[0], feats_B_0_upsampled), dim=1))
        fuse_feat2 = self.fuse2(torch.cat((feats_A[1], feats_B_1_upsampled), dim=1))
        fuse_feat3 = self.fuse3(torch.cat((feats_A[2], feats_B_2_upsampled), dim=1))
        fuse_feat4 = self.fuse4(torch.cat((feats_A[3], feats_B_3_upsampled), dim=1))

        # 特征上采样并进一步融合
        channel_matching_conv = nn.Conv2d(2048, 1024, kernel_size=1, bias=False).cuda()
        feats_B_4_upsampled = F.interpolate(feats_B[4], size=(feats_A[4].size(2), feats_A[4].size(3)), mode='bilinear', align_corners=False)
        feats_B_4_upsampled =channel_matching_conv(feats_B_4_upsampled)
        channel_matching_conv = nn.Conv2d(512, 1024, kernel_size=1, bias=False).cuda()
        feats_A_4_adjusted = channel_matching_conv(feats_A[4])
        up4 = self.up_concat4(fuse_feat4, feats_A_4_adjusted + feats_B_4_upsampled)  # 最深层特征直接相加
        up3 = self.up_concat3(fuse_feat3, up4)
        up2 = self.up_concat2(fuse_feat2, up3)
        up1 = self.up_concat1(fuse_feat1, up2)


        if self.up_conv != None:
            up1 = self.up_conv(up1)
        
        # 应用最终的卷积层
        final = self.final(up1)
        
        return final
    
    def freeze_backbone(self):
        # 冻结模态A的编码器参数
        for param in self.encoder_A.parameters():
            param.requires_grad = False
        # 冻结模态B的编码器参数
        for param in self.encoder_B.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        # 解冻模态A的编码器参数
        for param in self.encoder_A.parameters():
            param.requires_grad = True
        # 解冻模态B的编码器参数
        for param in self.encoder_B.parameters():
            param.requires_grad = True


class unetUp(nn.Module):   # 实现特征融合，实例分别为`up_concat4/3/2/1`，这些实例分别处理不同层次的特征融合和上采样。

    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs
    

# 具体的特征融合发生在`unetUp`类的`forward`方法中，通过将来自两个不同模态的特征图首先进行拼接，然后经过一系列的卷积和激活操作，最后通过上采样提升特征图的分辨率。

# 这个过程的关键是`torch.cat([inputs1, self.up(inputs2)], 1)`这行代码，它实现了特征融合的核心步骤
    # 首先，`self.up(inputs2)`通过双线性上采样将`inputs2`的分辨率提升至与`inputs1`相同；
    # 然后，`torch.cat`函数将两组特征沿着通道维度进行拼接。这样，来自不同模态的特征就被融合在了一起。
    # 随后的卷积层和ReLU激活函数进一步处理这些融合后的特征，为下一层的融合或最终的分类决策做准备。

# 每个`unetUp`实例代表不同的特征融合阶段，从深层特征到浅层特征，逐步将不同模态的信息合并，并逐渐提升特征图的分辨率，以便在网络的最终阶段进行精确的像素级分类。
# 这种逐层融合和上采样的策略是U-Net架构处理分割任务的关键特点之一，而在这个多模态版本中，它被用来整合来自不同模态的补充信息。
