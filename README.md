# 🔬 Mineral Image Classification

<div align="center">

**基于深度学习的矿物图像智能分类模型**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[特性](#-特性) • [快速开始](#-快速开始) • [模型架构](#-模型架构) • [使用示例](#-使用示例)

</div>

---

## 📖 项目简介

本项目是一个基于深度学习的矿物图像自动分类系统，能够准确识别和分类多种矿物类型。通过先进的卷积神经网络（CNN）和注意力机制，实现了高精度的矿物图像识别，为地质学、矿物学研究以及工业应用提供智能化的图像分析解决方案。

### 🎯 应用场景

- **地质勘探**：野外矿物快速识别与分类
- **矿物学研究**：自动化矿物样本分析
- **工业应用**：矿石分选与质量控制
- **教育科研**：矿物识别教学与研究工具

---

## ✨ 特性

### 🚀 核心功能

- **多模型架构支持**
  - 增强型基础CNN模型（深度网络设计）
  - CBAM注意力机制增强模型
  - 基于ResNet18的迁移学习模型

- **先进的注意力机制**
  - 通道注意力（Channel Attention）
  - 空间注意力（Spatial Attention）
  - 卷积块注意力模块（CBAM）

- **完整的数据处理流程**
  - 自动数据集拆分（训练集/验证集）
  - 丰富的数据增强策略
  - 智能数据预处理

- **专业的训练系统**
  - 自动学习率调度
  - 最佳模型自动保存
  - 训练过程可视化
  - 完整的训练历史记录

### 📊 数据集

- **7种矿物类别**：Biotite（黑云母）、Bornite（斑铜矿）、Chrysocolla（硅孔雀石）、Malachite（孔雀石）、Muscovite（白云母）、Pyrite（黄铁矿）、Quartz（石英）
- **5000+ 高质量图像**：涵盖多种拍摄条件和角度
- **自动数据划分**：支持自定义训练/验证集比例

---

## 🏗️ 模型架构

### 1. BasicCNN（增强型基础CNN）

深度卷积神经网络，采用多层卷积块设计：
- 5个卷积块，每个块包含3层卷积
- 4层全连接网络（2048→1024→512→7）
- He权重初始化优化
- 渐进式Dropout策略

### 2. CNNWithAttention（CBAM-CNN）

集成CBAM注意力机制的CNN模型：
- 在每个卷积块后应用注意力机制
- 通道注意力：关注重要特征通道
- 空间注意力：聚焦关键空间区域
- 提升模型对细节特征的感知能力

### 3. ResNetWithAttention（ResNet18+CBAM）

基于ResNet18的迁移学习模型：
- 利用ImageNet预训练权重（可选）
- 在ResNet特征提取层后集成CBAM
- 结合迁移学习与注意力机制的优势

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU训练，可选)
- 8GB+ RAM

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/YYYanZZZ/Deep-Learning-Based-Intelligent-Mineral-Image-Classification-Model.git
cd minet
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据集**
```bash
# 将矿物图像按类别放入 dataset/images/ 目录
# 目录结构示例：
# dataset/images/
#   ├── biotite/
#   ├── bornite/
#   ├── chrysocolla/
#   ├── malachite/
#   ├── muscovite/
#   ├── pyrite/
#   └── quartz/
```

4. **数据拆分与训练**
```bash
# 自动拆分数据集并开始训练
python main.py

# 或手动拆分数据
python split_data.py

# 然后训练模型
python train.py --model basic_cnn --epochs 50 --batch_size 32
```

---

## 💻 使用示例

### 基础训练

```bash
# 训练基础CNN模型
python train.py --model basic_cnn --epochs 50 --batch_size 32 --lr 0.001

# 训练带注意力机制的CNN模型
python train.py --model cnn_attention --epochs 50 --batch_size 32

# 训练ResNet+注意力模型（使用预训练权重）
python train.py --model resnet_attention --epochs 50 --pretrained
```

### 高级配置

```bash
# 自定义训练参数
python train.py \
    --model cnn_attention \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --image_size 256 \
    --save_dir ./my_checkpoints \
    --num_classes 7
```

### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型：`basic_cnn`, `cnn_attention`, `resnet_attention` | `basic_cnn` |
| `--epochs` | 训练轮数 | `50` |
| `--batch_size` | 批次大小 | `32` |
| `--lr` | 学习率 | `0.001` |
| `--image_size` | 输入图像尺寸 | `224` |
| `--pretrained` | 是否使用预训练权重（仅ResNet） | `False` |
| `--save_dir` | 模型保存目录 | `./checkpoints` |

---

## 📈 性能指标

### 模型性能对比

| 模型 | 参数量 | 验证准确率 | 特点 |
|------|--------|-----------|------|
| BasicCNN | ~15M | 55%+ | 深度网络，高容量 |
| CNNWithAttention | ~8M | 55%+ | 注意力增强，高效 |
| ResNetWithAttention | ~11M | 90%+ | 迁移学习，最佳性能 |

*注：实际性能取决于数据集质量和训练配置*

### 训练特性

- ✅ 自动学习率衰减（每10个epoch衰减10倍）
- ✅ 最佳模型自动保存
- ✅ 训练历史可视化
- ✅ 支持GPU加速训练
- ✅ 数据增强防止过拟合

---

## 🛠️ 技术栈

- **深度学习框架**：PyTorch 2.0+
- **计算机视觉**：Torchvision
- **数据处理**：NumPy, PIL
- **可视化**：Matplotlib
- **注意力机制**：CBAM (Convolutional Block Attention Module)

---

## 📁 项目结构

```
minet/
├── models.py              # 模型定义（BasicCNN, CNNWithAttention, ResNetWithAttention）
├── train.py               # 训练脚本
├── main.py                # 主程序入口
├── split_data.py          # 数据拆分工具
├── requirements.txt       # 依赖包列表
├── dataset/               # 数据集目录
│   ├── images/            # 原始图像（按类别分类）
│   ├── training/          # 训练集（自动生成）
│   └── validation/        # 验证集（自动生成）
└── checkpoints/           # 模型检查点（训练后生成）
```

---

## 🔬 技术亮点

### 1. CBAM注意力机制

实现了完整的CBAM（Convolutional Block Attention Module）模块：
- **通道注意力**：通过全局平均池化和最大池化提取通道特征
- **空间注意力**：结合通道维度的平均和最大池化，关注空间位置
- **顺序应用**：先通道后空间，最大化注意力效果

### 2. 数据增强策略

- 随机水平翻转（50%概率）
- 随机旋转（±15度）
- 颜色抖动（亮度、对比度、饱和度、色调）
- 随机仿射变换（平移）

### 3. 训练优化

- Adam优化器（weight_decay=1e-4）
- StepLR学习率调度器
- He权重初始化
- 渐进式Dropout策略

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！如果您想为项目做出贡献，请：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献方向

- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档改进
- 🎨 代码优化
- 🧪 测试用例

---

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 🙏 致谢

- 感谢 PyTorch 团队提供的优秀深度学习框架
- 感谢所有为开源社区做出贡献的开发者

---

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📧 提交 Issue
- 💬 开启 Discussion
- 🔗 项目链接：[GitHub Repository](https://github.com/YYYanZZZ/Deep-Learning-Based-Intelligent-Mineral-Image-Classification-Model)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star 支持一下！⭐**

Made with ❤️ by the Mineral Classification Team

</div>

