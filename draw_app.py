"""
draw_app.py
一个交互式“手写板”程序：打开一个窗口，你可以用鼠标在画布上手写数字，点击“识别”按钮后，
使用已训练好的 MNIST CNN 模型进行分类，并在界面上显示预测结果与置信度。

设计要点（尽量简单、便于理解）：
- 使用 Tkinter 创建窗口与画布（Canvas）
- 同步维护一张 PIL 图片（灰底/白底），捕捉鼠标轨迹并绘制到 PIL 中，以便送入模型推理
- 识别时做与训练一致的标准化（Normalize(mean=0.1307, std=0.3081)）
- 进行必要的图像预处理：灰度化、反色、居中裁剪与适当填充，然后缩放到 28×28
- 支持 CPU/GPU 自动选择（CUDA 优先）
使用方式：
    python MNIST_CNN/draw_app.py
"""


from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps, ImageTk

from model import SimpleCNN  # 同目录下的模型定义


# 与训练保持一致的标准化
def build_normalize_transform() -> transforms.Normalize:
    return transforms.Normalize(mean=(0.1307,), std=(0.3081,))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitDrawApp:
    """
    一个简单的 Tkinter 手写板应用。
    - 左键按下并移动即可在画布中画线
    - 点击“清空”按钮清除画布
    - 点击“识别”按钮执行模型推理并显示预测结果
    """

    def __init__(self, root: tk.Tk, model_path: str = None) -> None:
        self.root = root
        self.root.title("MNIST 手写板识别（SimpleCNN）")

        # 画布尺寸（较大，便于书写；随后会缩放到 28×28）
        self.canvas_size = 280
        self.bg_color = "white"  # 背景白色
        self.pen_color = "black"  # 画笔黑色（与 MNIST 风格相反，后续会做反色）
        self.pen_width = 16       # 线宽稍粗，便于形成可识别笔迹

        # Tkinter 画布
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg=self.bg_color)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # 同步维护的 PIL 图像，用于后续预处理与推理
        # 背景为白色（255），笔迹为黑色（0），最后会反色，使 MNIST 风格（黑底白字）
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # 绑定鼠标事件
        self.last_x, self.last_y = None, None
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 控件：识别、清空、设备显示
        self.btn_recognize = tk.Button(self.root, text="识别", command=self.on_recognize)
        self.btn_recognize.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.btn_clear = tk.Button(self.root, text="清空", command=self.on_clear)
        self.btn_clear.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.device_label = tk.Label(self.root, text=f"设备：{get_device()}")
        self.device_label.grid(row=1, column=2, padx=10, pady=5, sticky="e")

        # 结果显示区域
        self.result_var = tk.StringVar(value="请在画布中手写一个数字（0~9），然后点击“识别”")
        self.result_label = tk.Label(self.root, textvariable=self.result_var, font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        # 加载模型
        self.device = get_device()
        # 使用绝对路径，避免工作目录问题
        if model_path is None:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "mnist_cnn.pth")
        self.model = self.load_model(model_path, self.device)
        self.normalize = build_normalize_transform()

    def load_model(self, model_path: str, device: torch.device) -> SimpleCNN:
        """
        加载训练好的模型权重。
        """
        if not os.path.exists(model_path):
            messagebox.showerror(
                "模型未找到",
                f"未找到模型权重文件：{model_path}\n请先运行训练脚本：python MNIST_CNN/train.py"
            )
            raise FileNotFoundError(model_path)
        model = SimpleCNN().to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # 鼠标事件处理（在画布与 PIL 图像上同步绘制）
    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_mouse_drag(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # 在 Tkinter 画布上画线（显示）
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill=self.pen_color, width=self.pen_width, capstyle=tk.ROUND, smooth=True)
            # 在 PIL 图像上画线（用于推理）
            self.draw.line([(self.last_x, self.last_y), (x, y)], fill=0, width=self.pen_width)  # 黑色笔迹（0）
        self.last_x, self.last_y = x, y

    def on_button_release(self, event):
        self.last_x, self.last_y = None, None

    def on_clear(self):
        """
        清空画布与关联的 PIL 图像。
        """
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("画布已清空。请重新手写一个数字，然后点击“识别”")

    # 图像预处理：灰度化 -> 反色 -> 裁剪并居中 -> 填充 -> 缩放到 28x28 -> 转张量与标准化
    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """
        将画布中的 PIL 图像转换为模型需要的输入张量。
        步骤：
        1) 灰度化（已是 L 模式）
        2) 反色：将黑字白底变为白字黑底（更接近 MNIST 风格）
        3) 计算非背景像素的包围盒，裁剪并在黑底上居中摆放
        4) 缩放到 28×28
        5) ToTensor + Normalize(mean=0.1307, std=0.3081)
        """
        # 1) 灰度已满足（self.image 为 L 模式），此步可略
        img = pil_image

        # 2) 反色（白底黑字 -> 黑底白字）
        inv = ImageOps.invert(img)

        # 3) 计算包围盒并裁剪；若用户未绘制（没有有效笔迹），bbox 可能是整个图像或 None
        bbox = inv.getbbox()
        if bbox is None:
            # 没有有效笔迹，直接返回一个近似空白的输入（预测不稳定）
            # 这里提示用户重新书写
            raise ValueError("未检测到有效笔迹，请在画布上书写后再识别。")

        inv_cropped = inv.crop(bbox)  # 裁剪到有效区域
        w, h = inv_cropped.size
        max_side = max(w, h)

        # 4) 在黑底上居中摆放（加入适当边距，避免贴边）
        margin = int(0.15 * max_side)  # 留 15% 边距，使笔迹更居中更像 MNIST
        canvas_size = max_side + 2 * margin
        black_bg = Image.new("L", (canvas_size, canvas_size), color=0)  # 黑底
        paste_x = (canvas_size - w) // 2
        paste_y = (canvas_size - h) // 2
        black_bg.paste(inv_cropped, (paste_x, paste_y))

        # 5) 缩放到 28×28（双线性插值足够）
        resized = black_bg.resize((28, 28), Image.BILINEAR)

        # 6) ToTensor + Normalize
        to_tensor = transforms.ToTensor()  # [H,W] -> [1,H,W] in [0,1]
        tensor = to_tensor(resized)        # 形状 [1, 28, 28]
        tensor = self.normalize(tensor)    # 标准化
        return tensor

    def on_recognize(self):
        """
        执行推理：对当前画布图像进行预处理并送入模型，随后显示预测结果与置信度。
        """
        try:
            x = self.preprocess(self.image)       # [1,28,28]
        except ValueError as e:
            messagebox.showwarning("提示", str(e))
            return

        # 增加 batch 维度并移动到设备
        x = x.unsqueeze(0).to(self.device)        # [1,1,28,28]

        with torch.no_grad():
            logits = self.model(x)                # [1,10]
            probs = F.softmax(logits, dim=1).squeeze(0)  # [10]
            pred = int(probs.argmax().item())
            max_prob = float(probs[pred].item())

        # 在界面显示结果（预测类别与最大概率）
        self.result_var.set(f"预测结果：{pred}    置信度：{max_prob:.3f}\n各类别概率：{', '.join(f'{i}:{float(p):.2f}' for i,p in enumerate(probs.cpu().tolist()))}")

    def run(self):
        self.root.mainloop()


def main():
    # 创建 Tkinter 根窗口并启动应用
    root = tk.Tk()
    # 使用绝对路径，避免工作目录问题
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "mnist_cnn.pth")
    app = DigitDrawApp(root, model_path=model_path)
    app.run()


if __name__ == "__main__":
    main()
