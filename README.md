# MNIST CNN 手写数字识别项目

一个基于PyTorch的卷积神经网络（CNN）项目，用于识别手写数字0-9。本项目采用经典的MNIST数据集，提供完整的训练、推理和交互式识别功能，特别适合深度学习初学者学习和实践。

## 🚀 项目特色

- **完整的CNN实现**：从数据加载到模型训练、推理的全流程
- **交互式手写板**：支持鼠标书写数字并实时识别
- **优化训练策略**：包含数据增强、学习率调度等先进技术
- **模块化设计**：代码结构清晰，注释详尽，易于理解和扩展
- **多平台支持**：自动检测GPU/CPU，支持Windows/Linux/macOS

## 📁 项目结构

```
MNIST_CNN/
├── README.md              # 项目说明文档
├── model.py              # CNN模型定义
├── train.py              # 训练脚本（含数据增强）
├── infer.py              # 推理脚本
├── draw_app.py           # 交互式手写板应用
├── mnist_cnn.pth         # 预训练模型权重
├── requirements.txt      # 依赖包列表
└── data/                 # MNIST数据集（自动下载）
    └── MNIST/
        └── raw/          # 原始数据文件
```

## ⚙️ 环境要求

- Python 3.8+
- PyTorch 1.9+（CPU或GPU版本）
- torchvision
- matplotlib
- tkinter（用于手写板界面）

### 快速安装

```bash
# 安装依赖
pip install -r requirements.txt

# 或手动安装核心依赖
pip install torch torchvision matplotlib
```

> **注意**：Linux用户如需使用手写板功能，请安装tkinter支持：
> ```bash
> sudo apt-get install python3-tk
> ```

## 🎯 快速开始

### 1. 训练模型

```bash
# 基础训练（默认参数）
python train.py

# 高级训练（推荐参数）
python train.py --epochs 20 --batch_size 128 --lr 0.001
```

**训练特性**：
- 默认20个epoch，batch size 128
- 数据增强：随机旋转±10度，随机平移
- 学习率调度：每5个epoch学习率减半
- 自动保存最佳模型

### 2. 批量推理测试

```bash
# 测试集随机样本推理
python infer.py

# 自定义图片推理
python infer.py --image_path your_digit.png
```

### 3. 交互式手写识别

```bash
# 启动手写板应用
python draw_app.py
```

**手写板功能**：
- 鼠标书写数字
- 实时识别显示
- 概率分布可视化
- 一键清空画布

## 🧠 模型架构

本项目采用经典的CNN架构，专为MNIST数据集优化：

```python
# 输入：28×28灰度图像（1通道）
Conv2d(1→32, kernel=3, padding=1) + ReLU
MaxPool2d(2)  # 14×14
Conv2d(32→64, kernel=3, padding=1) + ReLU  
MaxPool2d(2)  # 7×7
Flatten()     # 64×7×7 = 3136
Linear(3136→128) + ReLU
Linear(128→10)  # 10类输出
```

## 📊 性能指标

- **测试准确率**：>98%（20个epoch训练后）
- **推理速度**：单张图片<10ms
- **模型大小**：~1.7MB

## 🔧 高级功能

### 数据增强
训练时自动应用数据增强技术：
- 随机旋转：±10度
- 随机平移：水平和垂直方向
- 标准化：使用MNIST统计参数

### 学习率调度
```python
# 每5个epoch学习率减半
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```

### 模型优化
- 自动GPU检测与使用
- 最佳模型保存机制
- 训练过程可视化

## 🐛 常见问题

### Q: 手写板无法启动？
**A**: 确保已安装tkinter：`sudo apt-get install python3-tk`（Linux）

### Q: 模型文件未找到？
**A**: 请先运行训练脚本生成模型文件：`python train.py`

### Q: 识别准确率低？
**A**: 尝试以下优化：
- 增加训练epoch：`--epochs 30`
- 调整学习率：`--lr 0.0005`
- 使用数据增强（默认已开启）

## 🎓 学习建议

### 初学者路径
1. 运行`train.py`理解训练流程
2. 使用`infer.py`测试模型效果
3. 体验`draw_app.py`交互功能
4. 阅读`model.py`理解CNN结构

### 进阶练习
- 修改网络结构（增加层数、通道数）
- 尝试不同的优化器和学习率策略
- 添加Dropout防止过拟合
- 实现训练曲线可视化

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发规范
- 保持代码注释清晰
- 遵循PEP8代码风格
- 添加适当的错误处理
- 更新相关文档

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- 感谢PyTorch团队提供的优秀深度学习框架
- 感谢MNIST数据集提供标准测试基准
- 感谢所有为开源社区贡献的开发者

---

**开始你的深度学习之旅吧！** 🚀

如有问题，欢迎在GitHub Issues中提出。