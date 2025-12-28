"""
主程序入口
用于运行完整的数据拆分和训练流程
"""

import os
import sys
from pathlib import Path

# 导入拆分脚本
from split_data import split_data, is_dir_empty

# 检查数据集目录结构
def check_dataset_structure():
    """检查数据集目录结构"""
    # 检查原始数据集目录（支持两种可能的路径）
    possible_src_dirs = [
        './dataset/mine/images',
        './dataset/images'
    ]
    
    src_dir = None
    for dir_path in possible_src_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            src_dir = dir_path
            break
    
    if src_dir is None:
        print("错误：找不到原始数据集目录！")
        print("请确保数据集位于以下目录之一：")
        for dir_path in possible_src_dirs:
            print(f"  - {dir_path}")
        return None
    
    print(f"找到数据集目录: {src_dir}")
    
    # 检查是否有子目录（类别目录）
    subdirs = [d for d in os.listdir(src_dir) 
               if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith('.')]
    
    if len(subdirs) == 0:
        print("警告：数据集目录下没有找到类别子目录！")
        return None
    
    print(f"找到 {len(subdirs)} 个类别目录: {subdirs}")
    return src_dir


def main():
    print("=" * 60)
    print("矿物图像分类项目 - 数据拆分和训练")
    print("=" * 60)
    
    # 步骤1: 检查数据集结构
    print("\n[步骤1] 检查数据集结构...")
    src_dir = check_dataset_structure()
    
    if src_dir is None:
        sys.exit(1)
    
    # 根据找到的源目录设置目标目录
    if 'mine' in src_dir:
        # 用户指定的目录结构
        training_dir = "./dataset/mine/training/"
        validation_dir = "./dataset/mine/validation/"
    else:
        # 如果使用的是 dataset/images，则使用 dataset/training 和 dataset/validation
        training_dir = "./dataset/training/"
        validation_dir = "./dataset/validation/"
    
    # 创建目录
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    # 步骤2: 数据拆分
    print("\n[步骤2] 拆分数据集...")
    train_dir_empty = is_dir_empty(training_dir)
    val_dir_empty = is_dir_empty(validation_dir)
    
    if train_dir_empty and val_dir_empty:
        print("训练集和验证集目录为空，开始拆分数据...\n")
        split_data(src_dir, training_dir, validation_dir,
                   test_dir=None, include_test_split=False, split_ratio=0.8)
        print("\n数据拆分完成！")
    else:
        if not train_dir_empty:
            print(f"⚠ 训练集目录 ({training_dir}) 不为空")
        if not val_dir_empty:
            print(f"⚠ 验证集目录 ({validation_dir}) 不为空")
        print("\n跳过数据拆分。如需重新拆分，请先清空训练集和验证集目录。")
    
    # 步骤3: 提示训练
    print("\n[步骤3] 训练模型")
    print("-" * 60)
    print("数据拆分完成！现在可以开始训练模型了。")
    print("\n训练命令示例：")
    print("  # 基础CNN模型")
    print("  python train.py --model basic_cnn --epochs 50 --batch_size 32")
    print("\n  # 带注意力机制的CNN模型")
    print("  python train.py --model cnn_attention --epochs 50 --batch_size 32")
    print("\n  # 基于ResNet的带注意力机制模型")
    print("  python train.py --model resnet_attention --epochs 50 --batch_size 32 --pretrained")
    print("\n更多选项请使用: python train.py --help")
    print("=" * 60)


if __name__ == "__main__":
    main()

