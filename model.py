"""
model.py
一个非常简洁的 CNN 模型（适合入门），用于 MNIST 手写数字识别（0~9）。

设计思路（尽量简单、但又具备 CNN 的关键要素）：
- 输入是 1×28×28 的灰度图（MNIST）
- 两个卷积 + ReLU + 池化 的组合，用于提取空间特征
- 展平后接两层全连接（全连接常用于分类任务的最后阶段）
- 最后一层输出 10 维 logits（未做 softmax，训练时交叉熵会自动做）

你可以把它理解为现代化、精简版的 LeNet 结构。
"""

# 让当前文件支持“未来版本”的语法特性：允许在类型注解中使用尚未定义的类名（如用 -> SimpleCNN 而无需加引号）
from __future__ import annotations

# PyTorch 核心库：提供张量（Tensor）及自动求导（autograd）等基础设施
import torch
# PyTorch 神经网络模块：封装了常用层（如 Conv2d、Linear）和容器（如 nn.Module）
import torch.nn as nn
# PyTorch 函数式接口：提供不保存权重的运算（如 relu、max_pool2d），常用于 forward 中
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    一个极简的卷积神经网络，用于 MNIST 分类任务。
    结构：
        - Conv2d(1 -> 32, kernel=3, padding=1) + ReLU
        - MaxPool2d(2)  # 尺寸：28x28 -> 14x14
        - Conv2d(32 -> 64, kernel=3, padding=1) + ReLU
        - MaxPool2d(2)  # 尺寸：14x14 -> 7x7
        - Flatten()
        - Linear(64*7*7 -> 128) + ReLU
        - Linear(128 -> 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # 卷积层：输入通道=1（灰度图），输出通道=32，卷积核=3x3，padding=1 保持空间尺寸不变
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 第二个卷积层：32 -> 64 通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 2x2 最大池化：将宽高各减半（下采样）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 展平后的特征维度：卷积+两次池化后，特征图为 64×7×7（因为 28 -> 14 -> 7）
        flattened_dim = 64 * 7 * 7

        # 全连接层：将特征映射到一个较小的隐藏维度，便于分类器学习
        self.fc1 = nn.Linear(in_features=flattened_dim, out_features=128)
        # 最终分类层：10 个类别（数字 0~9）
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # 可选：权重初始化（这里使用 PyTorch 默认即可，简单、稳妥）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        参数:
            x: [batch_size, 1, 28, 28] 的输入张量
        返回:
            logits: [batch_size, 10] 的输出（未经过 softmax）
        """
        # 卷积 -> ReLU -> 池化
        x = self.conv1(x)      # [B, 32, 28, 28]
        x = F.relu(x)
        x = self.pool(x)       # [B, 32, 14, 14]

        x = self.conv2(x)      # [B, 64, 14, 14]
        x = F.relu(x)
        x = self.pool(x)       # [B, 64, 7, 7]

        # 展平为二维：[B, 64*7*7]
        x = torch.flatten(x, start_dim=1)

        # 全连接分类器
        x = self.fc1(x)        # [B, 128]
        x = F.relu(x)
        logits = self.fc2(x)   # [B, 10]

        return logits


def count_parameters(model: nn.Module) -> int:
    """
    统计可训练参数量，便于了解模型大小（参数越多，模型越“大”）。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 当直接运行本文件时，做一个简单的自检
    model = SimpleCNN()
    dummy = torch.randn(2, 1, 28, 28)  # batch=2 的假数据
    out = model(dummy)
    print("输出形状（应为 [2, 10]）:", out.shape)
    print("可训练参数量:", count_parameters(model))
