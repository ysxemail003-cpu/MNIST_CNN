"""
infer.py
一个简单清晰的推理脚本：加载训练好的 CNN 权重，对 MNIST 测试集或你的自定义图片进行数字预测。

特点：
- 自动选择 CPU/GPU
- 若不给图片路径，则随机选取一张 MNIST 测试图片进行预测与可视化
- 若提供 --image_path，则对该图片（任意尺寸/彩色或灰度）做灰度化与缩放到 28x28，再预测
- 与训练一致的标准化（Normalize）保证预测效果稳定
- 代码注释详尽，帮助你理解每一步

使用：
    python MNIST_CNN/infer.py                # 随机取一张测试集图片进行预测和显示
    python MNIST_CNN/infer.py --image_path /path/to/your_digit.png  # 对你自己的图片进行预测
可选参数：
    python MNIST_CNN/infer.py --model_path MNIST_CNN/mnist_cnn.pth --data_root MNIST_CNN/data
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import SimpleCNN  # 相对导入同目录下的模型


def get_device() -> torch.device:
    """
    返回当前可用的设备（GPU 优先，CPU 备选）。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_normalize_transform() -> transforms.Normalize:
    """
    构建与训练时一致的标准化变换。
    使用 MNIST 的统计均值与方差：
        mean = 0.1307
        std  = 0.3081
    """
    return transforms.Normalize(mean=(0.1307,), std=(0.3081,))


def load_model(model_path: str, device: torch.device) -> SimpleCNN:
    """
    加载训练好的模型参数。如果找不到权重文件，提示先运行 train.py。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型权重文件：{model_path}\n"
            "请先运行训练脚本：python MNIST_CNN/train.py"
        )
    model = SimpleCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 推理模式（关闭 Dropout/BatchNorm 的训练行为等）
    return model


def preprocess_image_from_path(image_path: str) -> Tuple[torch.Tensor, Image.Image]:
    """
    从给定路径加载图片，并做如下处理：
    - 灰度化（convert('L')）
    - 缩放到 28x28（MNIST 的输入尺寸）
    - 转为张量（值范围 [0,1]，形状 [1, 28, 28]）
    返回：
      (tensor_28x28, pil_image_28x28)  # 供预测与可视化使用
    """
    img = Image.open(image_path)
    # 转灰度：'L' 表示 8-bit 像素，黑白
    img = img.convert('L')
    # 缩放到 MNIST 输入尺寸
    img = img.resize((28, 28), Image.BILINEAR)
    # 转为张量
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)  # [1, 28, 28], float32, [0,1]
    return tensor, img


def sample_random_test_image(data_root: str) -> Tuple[torch.Tensor, int]:
    """
    从 MNIST 测试集随机取一张图片。
    注意这里的 transform 仅使用 ToTensor()，这样便于直接显示原始图像；
    预测时我们会在此基础上再进行标准化（与训练保持一致）。
    返回：
      (tensor_0_1, label)
    """
    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),  # 仅做 ToTensor，便于展示
    )
    idx = random.randrange(len(test_set))
    img_tensor, label = test_set[idx]  # img_tensor: [1,28,28] in [0,1]
    return img_tensor, label


def predict(model: SimpleCNN, input_tensor: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor]:
    """
    对单张图片张量进行预测。
    参数：
        input_tensor: [1, 28, 28]，未标准化的张量（值在 [0,1]）
        device: 设备（CPU/GPU）
    过程：
        - 加 batch 维度 -> [1,1,28,28]
        - 标准化（与训练一致）
        - 前向传播得到 logits
        - softmax 得到每一类的概率
        - argmax 得到预测类别
    返回：
        (pred_class, probs)  # probs 形状 [10]，每一类的概率
    """
    normalize = build_normalize_transform()

    # 增加 batch 维度，且移动到设备上
    x = input_tensor.unsqueeze(0).to(device)  # [1, 1, 28, 28]
    # 对每张图进行标准化
    x = normalize(x)

    with torch.no_grad():
        logits = model(x)          # [1, 10]
        probs = F.softmax(logits, dim=1).squeeze(0)  # [10]
        pred = int(probs.argmax().item())
    return pred, probs.cpu()


def show_image_and_prediction(img_like, pred: int, probs: torch.Tensor, true_label: int | None = None) -> None:
    """
    可视化展示图片与预测结果。
    - img_like 可以是 PIL.Image（灰度）或张量 [1,28,28]（未标准化，便于显示）
    - 在标题里显示预测类别与概率最高值
    - 若提供 true_label，则一并显示真实标签
    """
    if isinstance(img_like, Image.Image):
        plt.imshow(img_like, cmap="gray")
    else:
        # 张量 [1,28,28] -> [28,28]
        plt.imshow(img_like.squeeze(0), cmap="gray")

    max_prob = float(probs[pred].item())
    title = f"预测：{pred}（置信度：{max_prob:.3f})"
    if true_label is not None:
        title += f" | 真实：{true_label}"
    plt.title(title)
    plt.axis("off")
    plt.show()

    # 同时打印整条概率向量，帮助你观察模型对每一类的信心
    probs_list = [float(p) for p in probs.tolist()]
    print("各类别概率（索引 0~9 为数字类别）：")
    for i, p in enumerate(probs_list):
        mark = "←" if i == pred else "  "
        print(f"  [{i}] {p:.3f} {mark}")


def main():
    parser = argparse.ArgumentParser(description="Infer a trained CNN on MNIST or a custom image.")
    parser.add_argument("--model_path", type=str, default="MNIST_CNN/mnist_cnn.pth", help="模型权重路径")
    parser.add_argument("--data_root", type=str, default="MNIST_CNN/data", help="MNIST 数据根目录（用于测试集随机取样）")
    parser.add_argument("--image_path", type=str, default=None, help="自定义图片路径（任意尺寸/彩色或灰度均可）")
    args = parser.parse_args()

    device = get_device()
    print("使用设备：", device)

    # 加载模型权重
    model = load_model(args.model_path, device)

    if args.image_path is not None:
        # 自定义图片推理
        tensor, pil_img = preprocess_image_from_path(args.image_path)
        pred, probs = predict(model, tensor, device)
        print(f"对自定义图片的预测结果：{pred}")
        show_image_and_prediction(pil_img, pred, probs, true_label=None)
    else:
        # MNIST 测试集随机采样
        os.makedirs(args.data_root, exist_ok=True)
        tensor, label = sample_random_test_image(args.data_root)
        pred, probs = predict(model, tensor, device)
        print(f"随机测试图片预测：{pred} | 真实标签：{label}")
        show_image_and_prediction(tensor, pred, probs, true_label=label)


if __name__ == "__main__":
    main()
