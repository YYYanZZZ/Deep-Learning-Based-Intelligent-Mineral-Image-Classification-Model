"""
矿物图像分类模型定义
包含基础CNN模型和带注意力机制的增强版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    """通道注意力模块 (Channel Attention Module)"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力模块 (Spatial Attention Module)"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# class BasicCNN(nn.Module):
#     """基础CNN模型"""
    
#     def __init__(self, num_classes=7):
#         super(BasicCNN, self).__init__()
#         # 第一个卷积块
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.25)
        
#         # 第二个卷积块
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.25)
        
#         # 第三个卷积块
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout2d(0.25)
        
#         # 第四个卷积块
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.dropout4 = nn.Dropout2d(0.25)
        
#         # 全连接层 - 使用自适应池化避免固定尺寸问题
#         self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(512, 512)
#         self.dropout5 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, num_classes)
    
#     def forward(self, x):
#         # 第一个卷积块
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)
#         x = self.dropout1(x)
        
#         # 第二个卷积块
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)
#         x = self.dropout2(x)
        
#         # 第三个卷积块
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = self.pool3(x)
#         x = self.dropout3(x)
        
#         # 第四个卷积块
#         x = F.relu(self.bn7(self.conv7(x)))
#         x = F.relu(self.bn8(self.conv8(x)))
#         x = self.pool4(x)
#         x = self.dropout4(x)
        
#         # 自适应池化
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)
        
#         # 全连接层
#         x = F.relu(self.fc1(x))
#         x = self.dropout5(x)
#         x = self.fc2(x)
        
#         return x
# class BasicCNN(nn.Module):
#     """第二次改进的基础CNN模型 - 增加模型容量以提升性能"""
    
#     def __init__(self, num_classes=7):
#         super(BasicCNN, self).__init__()
#         # 第一个卷积块 - 增加通道数
#         self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(96)
#         self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(96)
#         self.conv2_extra = nn.Conv2d(96, 96, kernel_size=3, padding=1)
#         self.bn2_extra = nn.BatchNorm2d(96)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.2)
        
#         # 第二个卷积块 - 增加通道数
#         self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(192)
#         self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(192)
#         self.conv4_extra = nn.Conv2d(192, 192, kernel_size=3, padding=1)
#         self.bn4_extra = nn.BatchNorm2d(192)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.2)
        
#         # 第三个卷积块 - 增加通道数
#         self.conv5 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(384)
#         self.conv6 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(384)
#         self.conv6_extra = nn.Conv2d(384, 384, kernel_size=3, padding=1)
#         self.bn6_extra = nn.BatchNorm2d(384)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout2d(0.2)
        
#         # 第四个卷积块 - 增加通道数
#         self.conv7 = nn.Conv2d(384, 768, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(768)
#         self.conv8 = nn.Conv2d(768, 768, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(768)
#         self.conv8_extra = nn.Conv2d(768, 768, kernel_size=3, padding=1)
#         self.bn8_extra = nn.BatchNorm2d(768)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.dropout4 = nn.Dropout2d(0.2)
        
#         # 第五个卷积块 - 新增更深的特征提取
#         self.conv9 = nn.Conv2d(768, 768, kernel_size=3, padding=1)
#         self.bn9 = nn.BatchNorm2d(768)
#         self.conv10 = nn.Conv2d(768, 768, kernel_size=3, padding=1)
#         self.bn10 = nn.BatchNorm2d(768)
#         # 使用全局平均池化而不是最大池化
#         self.pool5 = nn.AdaptiveAvgPool2d(1)
#         self.dropout5 = nn.Dropout2d(0.2)
        
#         # 全连接层 - 增加宽度，使用两层FC
#         self.fc1 = nn.Linear(768, 1024)
#         self.dropout6 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(1024, 512)
#         self.dropout7 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(512, num_classes)
    
#     def forward(self, x):
#         # 第一个卷积块
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn2_extra(self.conv2_extra(x)))
#         x = self.pool1(x)
#         x = self.dropout1(x)
        
#         # 第二个卷积块
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn4_extra(self.conv4_extra(x)))
#         x = self.pool2(x)
#         x = self.dropout2(x)
        
#         # 第三个卷积块
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = F.relu(self.bn6_extra(self.conv6_extra(x)))
#         x = self.pool3(x)
#         x = self.dropout3(x)
        
#         # 第四个卷积块
#         x = F.relu(self.bn7(self.conv7(x)))
#         x = F.relu(self.bn8(self.conv8(x)))
#         x = F.relu(self.bn8_extra(self.conv8_extra(x)))
#         x = self.pool4(x)
#         x = self.dropout4(x)
        
#         # 第五个卷积块
#         x = F.relu(self.bn9(self.conv9(x)))
#         x = F.relu(self.bn10(self.conv10(x)))
#         x = self.pool5(x)  # 全局平均池化
#         x = self.dropout5(x)
        
#         # 展平
#         x = x.view(x.size(0), -1)
        
#         # 全连接层
#         x = F.relu(self.fc1(x))
#         x = self.dropout6(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout7(x)
#         x = self.fc3(x)
        
#         return x
# class BasicCNN(nn.Module):
#     """第三次改进的基础CNN模型 - 使用残差连接和更合理的架构设计"""

#     def __init__(self, num_classes=7):
#         super(BasicCNN, self).__init__()
#         # 第一个卷积块 - 合理的通道数增长
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.1)  # 减少早期Dropout

#         # 第二个卷积块
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.15)

#         # 第三个卷积块 - 增加深度但保持合理通道数
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.conv6_extra = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn6_extra = nn.BatchNorm2d(256)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout2d(0.2)

#         # 第四个卷积块 - 更深的特征提取
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
#         self.conv8_extra = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8_extra = nn.BatchNorm2d(512)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.dropout4 = nn.Dropout2d(0.25)

#         # 第五个卷积块 - 在池化前增加特征提取能力
#         self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn9 = nn.BatchNorm2d(512)
#         self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn10 = nn.BatchNorm2d(512)
#         # 使用全局平均池化
#         self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
#         self.dropout5 = nn.Dropout2d(0.25)

#         # 全连接层 - 适度的宽度，三层FC
#         self.fc1 = nn.Linear(512, 1024)
#         self.dropout6 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, 512)
#         self.dropout7 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(512, num_classes)

#         # 权重初始化
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """使用He初始化来改善训练"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # 第一个卷积块
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)
#         x = self.dropout1(x)

#         # 第二个卷积块
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)
#         x = self.dropout2(x)

#         # 第三个卷积块 - 增加深度
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = F.relu(self.bn6_extra(self.conv6_extra(x)))
#         x = self.pool3(x)
#         x = self.dropout3(x)

#         # 第四个卷积块 - 更深特征提取
#         x = F.relu(self.bn7(self.conv7(x)))
#         x = F.relu(self.bn8(self.conv8(x)))
#         x = F.relu(self.bn8_extra(self.conv8_extra(x)))
#         x = self.pool4(x)
#         x = self.dropout4(x)

#         # 第五个卷积块 - 在池化前增强特征
#         x = F.relu(self.bn9(self.conv9(x)))
#         x = F.relu(self.bn10(self.conv10(x)))
#         x = self.adaptive_pool(x)  # 全局平均池化
#         x = self.dropout5(x)

#         # 展平
#         x = x.view(x.size(0), -1)

#         # 全连接层
#         x = F.relu(self.fc1(x))
#         x = self.dropout6(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout7(x)
#         x = self.fc3(x)

#         return x
# class BasicCNN(nn.Module):
#     """第四次改进的基础CNN模型 - 基于原始简单模型，保守增强"""

#     def __init__(self, num_classes=7):
#         super(BasicCNN, self).__init__()
#         # 第一个卷积块 - 保持原始设计
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.25)

#         # 第二个卷积块 - 保持原始设计
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.25)

#         # 第三个卷积块 - 保持原始设计
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout2d(0.25)

#         # 第四个卷积块 - 保持原始设计，但增加一层卷积以提升特征提取
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
#         self.conv8_extra = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8_extra = nn.BatchNorm2d(512)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.dropout4 = nn.Dropout2d(0.25)

#         # 全局平均池化
#         self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

#         # 全连接层 - 扩展容量以提升分类能力
#         self.fc1 = nn.Linear(512, 1024)
#         self.dropout5 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, 512)
#         self.dropout6 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(512, num_classes)

#         # 权重初始化 - 使用He初始化提升训练稳定性
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """使用He初始化来改善训练"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # 第一个卷积块
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)
#         x = self.dropout1(x)

#         # 第二个卷积块
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)
#         x = self.dropout2(x)

#         # 第三个卷积块
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = self.pool3(x)
#         x = self.dropout3(x)

#         # 第四个卷积块 - 增加一层卷积
#         x = F.relu(self.bn7(self.conv7(x)))
#         x = F.relu(self.bn8(self.conv8(x)))
#         x = F.relu(self.bn8_extra(self.conv8_extra(x)))
#         x = self.pool4(x)
#         x = self.dropout4(x)

#         # 全局平均池化
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)

#         # 全连接层 - 扩展为三层
#         x = F.relu(self.fc1(x))
#         x = self.dropout5(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout6(x)
#         x = self.fc3(x)

#         return x

class BasicCNN(nn.Module):
    """增强的基础CNN模型 - 大幅增加模型容量以提升性能到85%以上"""
    
    def __init__(self, num_classes=7):
        super(BasicCNN, self).__init__()
        # 第一个卷积块 - 增加深度
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_extra = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_extra = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)  # 减少早期Dropout
        
        # 第二个卷积块 - 增加深度
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4_extra = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4_extra = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.15)
        
        # 第三个卷积块 - 增加深度和宽度
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv6_extra = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6_extra = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        # 第四个卷积块 - 增加深度
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv8_extra = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8_extra = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # 第五个卷积块 - 新增，进一步提取特征
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv10_extra = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10_extra = nn.BatchNorm2d(512)
        # 使用全局平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout5 = nn.Dropout2d(0.25)
        
        # 全连接层 - 大幅扩展容量
        self.fc1 = nn.Linear(512, 2048)
        self.dropout6 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout7 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout8 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(512, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用He初始化来改善训练"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一个卷积块 - 3层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2_extra(self.conv2_extra(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块 - 3层
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn4_extra(self.conv4_extra(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块 - 3层
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn6_extra(self.conv6_extra(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 第四个卷积块 - 3层
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn8_extra(self.conv8_extra(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # 第五个卷积块 - 3层，在池化前
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn10_extra(self.conv10_extra(x)))
        x = self.adaptive_pool(x)
        x = self.dropout5(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层 - 4层，大幅扩展
        x = F.relu(self.fc1(x))
        x = self.dropout6(x)
        x = F.relu(self.fc2(x))
        x = self.dropout7(x)
        x = F.relu(self.fc3(x))
        x = self.dropout8(x)
        x = self.fc4(x)
        
        return x

class CNNWithAttention(nn.Module):
    """带注意力机制的CNN模型 (CBAM-CNN)"""
    
    def __init__(self, num_classes=7):
        super(CNNWithAttention, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention1 = CBAM(64, reduction=16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.attention2 = CBAM(128, reduction=16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 第三个卷积块
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.attention3 = CBAM(256, reduction=16)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 第四个卷积块
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.attention4 = CBAM(512, reduction=16)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # 全连接层 - 使用自适应池化避免固定尺寸问题
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention1(x)  # 应用注意力机制
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.attention2(x)  # 应用注意力机制
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.attention3(x)  # 应用注意力机制
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 第四个卷积块
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.attention4(x)  # 应用注意力机制
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        
        return x


class ResNetWithAttention(nn.Module):
    """基于ResNet的带注意力机制模型"""
    
    def __init__(self, num_classes=7, pretrained=False):
        super(ResNetWithAttention, self).__init__()
        # 加载ResNet18（可通过pretrained参数控制是否使用预训练权重）
        resnet = models.resnet18(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 添加注意力机制
        self.attention = CBAM(512, reduction=16)
        
        # 添加新的分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(model_name='basic_cnn', num_classes=7, **kwargs):
    """
    获取模型
    :param model_name: 模型名称 ('basic_cnn', 'cnn_attention', 'resnet_attention')
    :param num_classes: 分类数量
    :return: 模型实例
    """
    if model_name == 'basic_cnn':
        return BasicCNN(num_classes=num_classes)
    elif model_name == 'cnn_attention':
        return CNNWithAttention(num_classes=num_classes)
    elif model_name == 'resnet_attention':
        pretrained = kwargs.get('pretrained', False)  # 默认不使用预训练
        return ResNetWithAttention(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

