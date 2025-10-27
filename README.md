# MNIST_CNN 手写数字识别（PyTorch）

一个面向初学者的、目录清晰、注释详尽的子项目，用卷积神经网络（CNN）识别手写数字 0~9。数据集采用经典的 MNIST（28×28 的灰度手写数字图片）。

## 项目目标

- 通过一个不复杂但完整的 PyTorch 项目，帮你理解 CNN 的基本结构与训练流程。
- 提供清晰的目录与注释，让你能快速上手、轻松修改和扩展。
- 支持 CPU 与 GPU（如你本机安装了 CUDA 并且 `torch.cuda.is_available()` 为 True，训练会自动使用 GPU）。

## 目录结构

```
MNIST_CNN/
├─ README.md        # 使用说明与原理讲解（本文件）
├─ model.py         # 简单 CNN 模型定义（注释详尽）
├─ train.py         # 训练脚本（数据加载、训练循环、评估与保存）
├─ infer.py         # 推理脚本（从测试集或自定义图片做预测）
└─ requirements.txt # 依赖列表（可选，便于安装）
```

运行脚本会在 `MNIST_CNN/` 下自动下载 MNIST 数据到 `data/`（由 torchvision 负责），并保存模型权重到 `mnist_cnn.pth`。

## 环境准备

- Python 3.8+
- 建议安装 PyTorch 与 torchvision（CPU 或 GPU 版本均可）
- Matplotlib 用于可视化（可选）

你可以使用下面命令安装依赖（如已安装 PyTorch，可只安装其他库）：
```bash
pip install -r MNIST_CNN/requirements.txt
```

> 注意：不同系统/显卡对应的 PyTorch 安装方式可能不同，若 `pip install torch` 失败，请参考官方安装指引：https://pytorch.org/get-started/locally/

## 快速开始

1) 训练模型（默认训练 5 轮，batch size=64，学习率=1e-3）：
```bash
python MNIST_CNN/train.py
```

可选参数：
```bash
python MNIST_CNN/train.py --epochs 10 --batch_size 128 --lr 0.001
```

2) 在测试集上做一次推理（随机取一张测试图片并显示预测结果）：
```bash
python MNIST_CNN/infer.py
```

3) 对你自己的图片做预测（例如一张 28×28 的灰度 PNG）：
```bash
python MNIST_CNN/infer.py --image_path /path/to/your_digit.png
```

> 提示：如果你的图片不是 28×28、不是纯灰度，脚本会自动做灰度化与缩放。不用担心。

## 模型结构（model.py）

我们使用一个非常经典且简单的 CNN 结构，适合初学者理解：

- 输入：灰度图（1 个通道），大小 28×28
- 卷积块：
  - Conv2d(1→32, kernel=3, padding=1) + ReLU
  - MaxPool2d(2) 使特征图从 28×28 变为 14×14
  - Conv2d(32→64, kernel=3, padding=1) + ReLU
  - MaxPool2d(2) 使特征图从 14×14 变为 7×7
- 分类器：
  - Flatten 展平（64×7×7 → 3136）
  - Linear(3136 → 128) + ReLU
  - Linear(128 → 10) 输出 10 类的 logits（未归一化分数）

这类似于精简版的 LeNet-5 思想，但更加现代与简洁。

## 训练流程（train.py）

- 设置随机种子，保证一定的可复现性。
- 使用 `torchvision.datasets.MNIST` 自动下载并加载数据集。
- 数据预处理：
  - `ToTensor()`：把 PIL 图片转为张量，像素值缩放到 [0,1]
  - `Normalize(mean=0.1307, std=0.3081)`：用 MNIST 统计值做标准化（提升训练稳定性）
- 使用 `DataLoader` 打包为批次（batch）并打乱（shuffle）。
- 定义损失函数 `CrossEntropyLoss` 与优化器 `Adam`。
- 标准训练循环：
  1. 前向传播得到预测 `logits`
  2. 计算损失 `loss`
  3. 反向传播 `loss.backward()`
  4. 更新参数 `optimizer.step()`
- 每个 epoch 结束后在测试集上计算准确率，观察训练效果。
- 训练结束保存权重到 `MNIST_CNN/mnist_cnn.pth`

## 推理与可视化（infer.py）

- 加载 `model.py` 中同样结构的模型，并加载训练好的权重。
- 若提供 `--image_path`，则对该图片做灰度化、缩放到 28×28、转张量与同样的标准化，然后预测。
- 若未提供图片路径，则从 MNIST 测试集中随机取一张进行预测。
- 会打印预测的数字类别，并可视化显示图片与预测结果。

## 常见问题与建议

- 设备选择：脚本会自动检测 GPU（CUDA），若不可用则用 CPU。CPU 训练也可行，只是会略慢。
- num_workers：为了避免某些平台上的多进程数据加载问题，脚本里默认 `num_workers=2`（你也可以改成 0 来更稳妥）。
- 标准化：推理时必须与训练时用一致的标准化参数，否则预测可能不准。
- 尝试修改：你可以尝试增减卷积层数、更改通道数、使用不同优化器或学习率，观察对准确率的影响。

## 你可以做的扩展练习

- 在 `train.py` 里添加训练损失与准确率的曲线绘制。
- 在 `infer.py` 打印每一类的概率（softmax 后），并画成柱状图。
- 改用更深一些的网络（例如增加一层卷积与 Dropout），对比效果与过拟合情况。

祝学习顺利！如果你已经准备好，直接运行训练与推理脚本，看看你的第一版 CNN 能达到怎样的准确率吧。

## 交互式手写板（draw_app.py）

你可以通过一个可视化的“手写板”来用鼠标书写数字并进行识别。

运行：
```bash
python MNIST_CNN/draw_app.py
```

使用说明：
- 在画布用鼠标左键书写，松开后可继续书写。
- 点击“识别”按钮后，界面会显示预测类别与各类概率。
- 点击“清空”可重置画布，重新书写。

识别原理小结（与训练一致的预处理）：
- 程序将白底黑字自动反色为黑底白字，使风格接近 MNIST。
- 自动裁剪有效笔迹区域，并在黑底上居中摆放；随后缩放到 28×28。
- 使用 Normalize(mean=0.1307, std=0.3081) 做标准化，与训练保持一致。

常见问题：
- 无法弹窗或提示 `No module named '_tkinter'`：Linux 下需要安装 Tk 支持（系统包），例如在 Ubuntu/Debian 运行：
  ```bash
  sudo apt-get install python3-tk
  ```
  注意：Tkinter 是系统包，不通过 pip 安装；即使在虚拟环境中，也需要系统层面的 `python3-tk`。
- 未找到模型权重：请先运行训练脚本生成 `MNIST_CNN/mnist_cnn.pth`。

书写小建议：
- 使用较粗的笔迹并尽量居中书写，避免过细或贴边（更接近 MNIST）。
- 背景保持白色，笔迹为黑色即可（程序会自动反色）。
