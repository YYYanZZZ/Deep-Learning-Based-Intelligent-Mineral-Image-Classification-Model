"""
数据拆分脚本
将原始数据集按比例拆分为训练集和验证集
"""

import os
import sys
import random
from pathlib import Path
from shutil import copyfile


def is_dir_empty(path):
    """检查目录是否为空"""
    return not any(Path(path).iterdir())


def split_data(src_dir, training_dir, validation_dir, test_dir=None,
               include_test_split=False, split_ratio=0.8):
    """
    将源数据按比例复制到 training 目录和 validation 目录（支持子目录结构）
    :param src_dir: 源数据根目录
    :param training_dir: 训练集输出目录
    :param validation_dir: 验证集输出目录
    :param test_dir: 测试集输出目录（可选）
    :param include_test_split: 是否分割测试集
    :param split_ratio: 将源数据按比例复制到 training 目录和 validation 目录，默认 8 : 2 比例，默认有测试集，只分割出验证集
    """
    # 遍历所有子目录，保持目录结构
    for root, dirs, files in os.walk(src_dir):
        """
        root: 当前正在遍历的目录的绝对/相对路径（取决于 src_dir 的格式）
        dirs: root 目录下的所有子目录名称列表（仅一级，不递归）
        files: root 目录下的所有文件名称列表（仅一级，不含子目录里的文件）
        """
        # 定义允许的文件扩展名（不区分大小写）
        ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

        # 跳过无文件的目录
        if not files:
            continue

        # 过滤掉空文件并获取有效文件列表
        valid_files = []
        for file in files:
            file_path = os.path.join(root, file)
            # 获取文件扩展名（转为小写，兼容大写扩展名如.JPG、.PNG）
            file_ext = os.path.splitext(file)[1].lower()
            if os.path.getsize(file_path) > 0 and file_ext in ALLOWED_EXTENSIONS:  # 检查文件大小是否大于 0
                valid_files.append(file)

        # 如果当前目录没有有效文件，跳过
        if not valid_files:
            continue

        # 计算相对路径，用于在目标目录创建相同的子目录结构
        rel_path = os.path.relpath(root, src_dir)
        # print(root, src_dir, rel_path)

        # 创建训练集对应目录
        train_subdir = os.path.join(training_dir, rel_path)
        os.makedirs(train_subdir, exist_ok=True)

        # 创建验证集对应目录
        val_subdir = os.path.join(validation_dir, rel_path)
        os.makedirs(val_subdir, exist_ok=True)

        # 如果需要测试集，创建测试集对应目录
        test_subdir = None
        if include_test_split and test_dir:
            test_subdir = os.path.join(test_dir, rel_path)
            os.makedirs(test_subdir, exist_ok=True)

        # 随机打乱文件
        shuffled_files = random.sample(valid_files, len(valid_files))

        # 计算分割点
        train_split = int(split_ratio * len(shuffled_files))
        train_files = shuffled_files[:train_split]
        if include_test_split:
            # 剩余部分平分给验证集和测试集
            val_test_split = train_split + (len(shuffled_files) - train_split) // 2
            val_files = shuffled_files[train_split:val_test_split]
            test_files = shuffled_files[val_test_split:]
        else:
            # 所有剩余文件都作为验证集
            print(f"Split validation only for directory: {rel_path}")
            val_files = shuffled_files[train_split:]
            test_files = []

        # 复制训练集文件
        for file in train_files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(train_subdir, file)
            copyfile(src_file_path, dst_file_path)

        # 复制验证集文件
        for file in val_files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(val_subdir, file)
            copyfile(src_file_path, dst_file_path)

        # 复制测试集文件（如果需要）
        if include_test_split and test_files:
            for file in test_files:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(test_subdir, file)
                copyfile(src_file_path, dst_file_path)

        # 打印当前目录的分割信息
        print(f"Processed directory: {rel_path}")
        print(f" - Training files: {len(train_files)}")
        print(f" - Validation files: {len(val_files)}")
        if include_test_split:
            print(f" - Test files: {len(test_files)}")

    print("\nSplit successful! All directories processed.")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)

    # 自动检测数据集路径（支持两种可能的路径结构）
    possible_src_dirs = [
        './dataset/mine/images',
        './dataset/images'
    ]
    
    SRC_MINE_DIR = None
    for dir_path in possible_src_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            SRC_MINE_DIR = dir_path
            break
    
    if SRC_MINE_DIR is None:
        print("错误：找不到原始数据集目录！")
        print("请确保数据集位于以下目录之一：")
        for dir_path in possible_src_dirs:
            print(f"  - {dir_path}")
        sys.exit(1)
    
    # 根据找到的源目录设置目标目录
    if 'mine' in SRC_MINE_DIR:
        TRAINING_DIR = "./dataset/mine/training/"
        VALIDATION_DIR = "./dataset/mine/validation/"
        TEST_DIR = "./dataset/mine/test/"
    else:
        TRAINING_DIR = "./dataset/training/"
        VALIDATION_DIR = "./dataset/validation/"
        TEST_DIR = "./dataset/test/"
    
    print(f"使用数据集目录: {SRC_MINE_DIR}")
    print(f"训练集目录: {TRAINING_DIR}")
    print(f"验证集目录: {VALIDATION_DIR}")

    # 创建目录（如果不存在）
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    # 检查训练集和验证集目录是否为空
    train_dir_empty = is_dir_empty(TRAINING_DIR)
    val_dir_empty = is_dir_empty(VALIDATION_DIR)

    if train_dir_empty and val_dir_empty:
        print("Training and Validation directories are empty. Starting data split...\n")
        split_data(SRC_MINE_DIR, TRAINING_DIR, VALIDATION_DIR,
                   test_dir=None, include_test_split=False, split_ratio=0.8)
    else:
        # 提示目录不为空的原因
        if not train_dir_empty:
            print(f"❌ Training directory ({TRAINING_DIR}) is not empty!")
        if not val_dir_empty:
            print(f"❌ Validation directory ({VALIDATION_DIR}) is not empty!")
        print("\nSkip data split. Please empty the directories first if you want to re-run the split.")

