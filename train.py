"""
train.py
一个清晰易懂的训练脚本，使用简单的 CNN 在 MNIST 数据集上进行分类训练。

特点：
- 自动下载 MNIST 数据到 MNIST_CNN/data 目录
- 支持 CPU/GPU 自动选择（如 torch.cuda.is_available() 为 True 则用 GPU）
- 结构与流程尽量简单，注释详尽，便于初学者理解
- 训练结束自动保存模型权重到 MNIST_CNN/mnist_cnn.pth

使用：
    python MNIST_CNN/train.py
可选参数：
    python MNIST_CNN/train.py --epochs 10 --batch_size 128 --lr 0.001 --num_workers 2
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN  # 直接相对导入同目录下的模型


def get_device() -> torch.device:
    """
    返回当前可用的设备（GPU 优先，CPU 备选）。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    """
    设置随机种子，增强结果的可复现性。
    注意：完全复现还会受到硬件、库版本等影响，这里做的是“尽量一致”。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 为了更稳定的结果（牺牲一些速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    use_data_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练集与测试集的 DataLoader。

    - 数据预处理：
      * ToTensor(): 将 [0,255] 像素值转换到 [0,1] 并变为张量
      * Normalize(mean=0.1307, std=0.3081): 用 MNIST 的统计值标准化，提升稳定性
    - 数据增强（仅训练集）：随机旋转、平移、缩放
    - datasets.MNIST: 若本地无数据，会自动下载到 data_root

    返回：
      (train_loader, test_loader)
    """
    # 测试集使用基本预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])
    
    # 训练集使用数据增强
    if use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),  # 随机旋转 ±10度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ])
    else:
        train_transform = test_transform

    train_set = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,   # 训练集通常需要打乱
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # GPU 上可提升拷贝效率
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    在给定的 DataLoader 上评估准确率（accuracy）。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)               # [B, 10]
            preds = logits.argmax(dim=1)         # 取最大值所在的类别索引
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_index: int,
) -> float:
    """
    训练一个 epoch，并返回该 epoch 的平均损失（loss）。
    """
    model.train()
    running_loss = 0.0
    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        logits = model(images)

        # 计算损失（交叉熵适用于分类任务，内部会做 LogSoftmax）
        loss = criterion(logits, targets)

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每隔若干批次打印一次进度与当前批次损失，便于观察训练过程
        if batch_idx % 100 == 0 or batch_idx == total_batches:
            print(f"[Epoch {epoch_index} | Batch {batch_idx}/{total_batches}] loss = {loss.item():.4f}")

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮次（越大可能越好，但更耗时）")
    parser.add_argument("--batch_size", type=int, default=128, help="每批次的样本数量（受显存影响）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（优化器的步长）")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 的工作线程数（若报错可改为 0）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（增强可复现性）")
    parser.add_argument("--data_root", type=str, default="MNIST_CNN/data", help="数据根目录（MNIST 自动下载到此处）")
    parser.add_argument("--save_path", type=str, default="MNIST_CNN/mnist_cnn.pth", help="模型权重保存路径")
    args = parser.parse_args()

    # 1) 随机种子与设备
    seed_everything(args.seed)
    device = get_device()
    print("使用设备：", device)

    # 2) 数据加载
    os.makedirs(args.data_root, exist_ok=True)  # 保险起见，确保目录存在
    train_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 3) 模型、损失与优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器 - 每5个epoch降低学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 4) 训练循环
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch,
        )
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}: avg_train_loss = {avg_train_loss:.4f}, test_acc = {test_acc:.4f}")

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 简单的"最好模型"记录（以测试准确率为指标）
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"→ 提升！已保存当前最好模型到：{args.save_path}（test_acc = {best_test_acc:.4f}）")
        
        print(f"当前学习率：{current_lr:.6f}")

    # 5) 训练结束再保存一份最终权重（覆盖或与最好模型一致）
    torch.save(model.state_dict(), args.save_path)
    print(f"训练完成，最终模型已保存到：{args.save_path}")
    print(f"最好测试准确率：{best_test_acc:.4f}")


if __name__ == "__main__":
    main()
