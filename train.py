"""
训练脚本
用于训练矿物图像分类模型
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from models import get_model
import numpy as np


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 50 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train(model, train_loader, val_loader, num_epochs, device, lr=0.001, 
          save_dir='./checkpoints', model_name='mineral_classifier'):
    """训练模型"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"开始训练，共 {num_epochs} 个epochs")
    print(f"设备: {device}")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
            print(f"✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        
        # 每个epoch保存一次
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth'))
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch [{epoch}/{num_epochs}] ({elapsed_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
    
    print(f"\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history
    }, os.path.join(save_dir, f'{model_name}_final.pth'))
    
    return history


def plot_training_history(history, save_path='./training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制准确率
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图表已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='训练矿物图像分类模型')
    parser.add_argument('--model', type=str, default='basic_cnn',
                        choices=['basic_cnn', 'cnn_attention', 'resnet_attention'],
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_classes', type=int, default=7, help='分类数量')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='训练集目录（默认自动检测）')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='验证集目录（默认自动检测）')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--image_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='是否使用预训练权重（仅对resnet_attention有效）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 自动检测数据集目录（如果未指定）
    if args.train_dir is None or args.val_dir is None:
        possible_train_dirs = [
            './dataset/mine/training',
            './dataset/training'
        ]
        possible_val_dirs = [
            './dataset/mine/validation',
            './dataset/validation'
        ]
        
        train_dir = args.train_dir
        val_dir = args.val_dir
        
        if train_dir is None:
            for dir_path in possible_train_dirs:
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    train_dir = dir_path
                    break
        
        if val_dir is None:
            for dir_path in possible_val_dirs:
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    val_dir = dir_path
                    break
        
        if train_dir is None or val_dir is None:
            print("错误：找不到训练集或验证集目录！")
            print("请使用 --train_dir 和 --val_dir 参数指定目录，或确保数据已拆分。")
            return
    else:
        train_dir = args.train_dir
        val_dir = args.val_dir
    
    # 加载数据集
    print(f"\n加载数据集...")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"类别数量: {len(train_dataset.classes)}")
    print(f"类别名称: {train_dataset.classes}")
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = get_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 训练
    model_name = f"{args.model}_mineral_classifier"
    history = train(model, train_loader, val_loader, args.epochs, device, 
                    lr=args.lr, save_dir=args.save_dir, model_name=model_name)
    
    # 绘制训练历史
    plot_training_history(history, save_path=f'./{model_name}_history.png')
    
    print("\n所有任务完成！")


if __name__ == '__main__':
    main()

