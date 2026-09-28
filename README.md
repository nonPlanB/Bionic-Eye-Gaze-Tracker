# Bionic Eye Gaze Tracker (仿生墨水屏视线追踪系统)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Arduino / ESP32](https://img.shields.io/badge/Platform-ESP32%20%7C%20Arduino-teal.svg?logo=espressif&logoColor=white)](https://www.espressif.com/)
[![YOLOv8](https://img.shields.io/badge/AI%20Model-YOLOv8-orange.svg?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![TensorRT](https://img.shields.io/badge/Acceleration-NVIDIA%20TensorRT-green.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![BLE](https://img.shields.io/badge/Protocol-BLE%20GATT-lightgrey.svg?logo=bluetooth&logoColor=white)](https://www.bluetooth.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

一个将**实时机器视觉**与**微型嵌入式电子墨水屏**深度融合的仿生眼球凝视同步系统。

本项目通过摄像头实时捕捉人眼部动态，采用定制微调的 **YOLOv8** 目标检测引擎、针对眼球跳视（Saccade）优化的 **8 状态卡尔曼滤波**与非线性空间多项式映射，将注视点坐标经由 **BLE (蓝牙低功耗)** 实时无线同步至 ESP32，并在 1.54 英寸电子墨水屏上通过三图层点阵合成与高速局部刷新算法呈现拟真、生动的眼球注视效果。

---

## 目录
- [系统架构](#系统架构)
- [核心功能与特性](#核心功能与特性)
- [硬件配置与引脚定义](#硬件配置与引脚定义)
- [技术栈](#技术栈)
- [项目目录结构](#项目目录结构)
- [安装与运行指南](#安装与运行指南)
  - [1. 硬件固件烧录 (ESP32)](#1-硬件固件烧录-esp32)
  - [2. 上位机环境配置](#2-上位机环境配置)
  - [3. 视线空间九点标定](#3-视线空间九点标定)
  - [4. 启动实时追踪与控制](#4-启动实时追踪与控制)
- [工具链与高级使用](#工具链与高级使用)
  - [模型训练与格式转换](#模型训练与格式转换)
  - [墨水屏点阵素材转换](#墨水屏点阵素材转换)
  - [BLE 手动调试](#ble-手动调试)
- [配置说明](#配置说明)
- [许可证](#许可证)

---

## 系统架构

```
 [ 摄像头输入 (640x480) ]
            │
            ▼
 [ YOLOv8 目标检测引擎 ] ──(PT / ONNX / TensorRT FP16)
            │
            ▼
 [ 动态门限滤波与状态估计 ] ──(8 状态卡尔曼滤波 + EWMA 加权平滑)
            │
            ▼
 [ 非线性多项式映射引擎 ] ──(二阶曲面拟合，约束至物理视界)
            │
            ▼
 [ 异步 BLE GATT 无响应传输 ] ──(低延迟无阻塞通道 "X,Y")
            │
            ▼ (Wireless)
 [ ESP32 蓝牙从机 (Eyes.ino) ]
            │
            ▼
 [ 动态三图层位运算合成 ] ──(底层眼白 + 中层移动瞳孔 + 顶层眼眶掩码)
            │
            ▼ (SPI 10MHz)
 [ 200x200 电子墨水屏 (EPD) 局部高速刷新 ]
```

---

## 核心功能与特性

1. **高精度实时眼球检测**：
   - 专用单类别 `eye` 定制训练模型，验证集 mAP@0.5 达 **99.5%**，mAP@0.5:0.95 达 **97.2%**。
   - 生产级多后端兼容：支持 PyTorch（CPU/CUDA）、ONNX Runtime，并支持编译为 **TensorRT Engine (FP16)** 获得极致帧率。
2. **抗噪运动滤波与瞬态预测**：
   - **8 状态恒速卡尔曼滤波**：自适应抗跳变门限（Gating Check），有效过滤反光、阴影误检。
   - **眨眼/遮挡预测补全**：短时间闭眼或反光丢失时沿物理速度外推；超时自动平滑回中（Auto-Center）。
   - **EWMA 指数滑动窗口**：消除墨水屏高频抖动毛刺。
3. **半自动对角九点空间标定**：
   - 支持在视频画面中点击两个对角点自动生成 $3 \times 3$ 标定阵列，并支持动态缩放间距。
   - 集成注视稳定性门限采样器，通过最小二乘法拟合双向二次多项式映射矩阵。
4. **低延迟 BLE GATT 管道**：
   - 采用 `Write Without Response` 模式，避免传统阻塞式等待，最小发送间隔达 50ms。
   - 具备断线自动扫描重连机制与定时保活心跳（Keep-Alive）。
5. **墨水屏三图层像素合成与刷新**：
   - 采用底层（固定眼白）、中层（平移圆形瞳孔）、顶层（眼眶开孔掩码遮罩）三重合成流水线。
   - 字节级并行位运算（Bitwise Masking），大幅减少 MCU 运算负载。
   - 具备局部刷新防残影保护机制（每累计 30 次局部刷新自动触发一次全局刷新消影）。

---

## 硬件配置与引脚定义

### 硬件物料
- **主控芯片**：ESP32 开发板 (如 ESP32-WROOM-32 / NodeMCU-32S / ESP32-M1)
- **显示屏幕**：1.54 英寸黑白电子墨水屏（200×200 分辨率，EPD_W21 / SSD1681 驱动系列）
- **视觉前端**：USB 广角/近距无畸变摄像头或笔记本自带摄像头

### ESP32 与墨水屏 SPI 接线映射

| 墨水屏引脚 (EPD Pin) | ESP32 GPIO | 说明 |
| :--- | :--- | :--- |
| **SCLK / CLK** | **GPIO 18** | SPI 时钟总线 |
| **MOSI / DIN** | **GPIO 23** | SPI 主出从入数据线 |
| **CS** | **GPIO 27** | 片选信号 (低电平有效) |
| **DC** | **GPIO 14** | 数据 / 命令切换引脚 (高: 数据 / 低: 命令) |
| **RST / RES** | **GPIO 12** | 硬件复位引脚 |
| **BUSY** | **GPIO 13** | 墨水屏忙状态指示 (低电平空闲) |
| **VCC** | **3.3V** | 电源正极 |
| **GND** | **GND** | 电源地 |

---

## 技术栈

- **嵌入式与硬件**：C++ / Arduino Core for ESP32, SPI, BLE GATT Server, E-Paper LUT Controller
- **机器视觉与深度学习**：Python 3.10+, PyTorch, Ultralytics YOLOv8, NVIDIA TensorRT, OpenCV
- **通信与异步调度**：Bleak (Asyncio BLE), NumPy, Pillow (Floyd-Steinberg 误差扩散算法)

---

## 项目目录结构

```text
├── config/                          # 全局配置文件
│   ├── dataset_single_class.yaml    # YOLO 数据集路径与类别配置
│   ├── mapping.json                 # 视线像素坐标到屏幕位移的多项式系数
│   ├── settings.json                # 上位机硬件、模型、通信全局配置文件
│   └── train_config.json            # 模型超参数配置
├── datasets/                        # 数据集规范目录 (已分划 train/valid/test)
├── images/                          # 墨水屏素材与转换工具
│   ├── convert.py                   # Floyd-Steinberg 抖动与 C 数组打包脚本
│   ├── tongkong.png / yanbai.png    # 原始素材
│   └── ...
├── inference/
│   └── detector.py                  # 封装的生产级眼部目标检测推理类
├── models/                          # 训练就绪的权重文件 (.pt / .engine)
├── runs/                            # 模型训练、验证指标曲线与权重输出
├── scripts/                         # 上位机执行脚本
│   ├── calibrate.py                 # 九点对角交互式视线标定程序
│   ├── run.py                       # 实时视觉追踪与 BLE 联动主程序
│   └── test_ble_esp32.py            # BLE 链路手动交互调试工具
├── src/artificial_eye/              # 上位机核心模块
│   ├── config.py                    # 配置加载器
│   ├── core.py                      # 卡尔曼滤波、加权平滑与坐标映射器
│   ├── hardware.py                  # Bleak 异步非阻塞 BLE 发射器
│   └── tracker.py                   # 视觉采集与追踪主流水线
├── tools/                           # 数据集管理与训练工具链
│   ├── benchmark.py                 # 多后端 (PT/ONNX/Engine) 性能压测
│   ├── capture_samples.py           # 样本快速采集工具
│   ├── export.py                    # TensorRT Engine / ONNX 导出及对齐验证
│   ├── prepare_dataset.py           # 标签清洗与数据泄露检查
│   ├── split_dataset.py             # 训练/验证/测试集自动划分
│   └── train.py                     # YOLOv8 训练主程序
├── Display_EPD_W21*                 # 墨水屏硬件驱动源码
├── Eyes.ino                         # ESP32 固件主文件
└── tongkong.h / yanbai.h / yankuang.h # 预转换的墨水屏 1-bit 点阵素材头文件
```

---

## 安装与运行指南

### 1. 硬件固件烧录 (ESP32)

1. 安装 [Arduino IDE](https://www.arduino.cc/en/software)，并在首选项中添加 ESP32 开发板支持包：
   ```text
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
2. 安装 `ESP32 by Espressif Systems` 开发板依赖。
3. 打开 `Eyes.ino`，确保同级目录下包含：
   - `Display_EPD_W21.cpp` / `.h`
   - `Display_EPD_W21_spi.cpp` / `.h`
   - `tongkong.h`, `yanbai.h`, `yankuang.h`
4. 开发板选择 `ESP32 Dev Module`，编译并烧录至 ESP32。
5. 烧录完成后打开串口监视器（波特率 `115200`），若提示 `BLE advertising started.` 即代表固件就绪。

---

### 2. 上位机环境配置

推荐使用 Conda 创建独立的 Python 虚拟环境：

```bash
# 创建并激活 Python 3.10 环境
conda create -n eye_tracker python=3.10 -y
conda activate eye_tracker

# 安装 PyTorch (建议配合本机 CUDA 版本)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖库
pip install ultralytics opencv-python bleak numpy pillow
```

*(可选)* 若需使用 NVIDIA TensorRT FP16 加速，请确保安装与系统 CUDA 对应的 `tensorrt` 模块。

---

### 3. 视线空间九点标定

为了使人眼在屏幕上的注视移动准确映射到仿生眼球的开孔几何空间内，需在首次运行时执行空间标定：

```bash
python scripts/calibrate.py --samples 30 --spacing 1.0
```

#### 标定操作流程：
1. **框选对角点**：使用鼠标左键依次点击摄像头画面中目标注视区域的两个对角（如**左上角**与**右下角**）。
2. **预览并微调网格**：
   - 键盘按下 `+` 或 `=`：等比扩大 9 点分布间距。
   - 键盘按下 `-` 或 `_`：等比缩小 9 点分布间距。
   - 键盘按下 `R`：重置并重新选择对角点。
   - 键盘按下 `空格键`：确认网格形状，进入凝视标定阶段。
3. **依次凝视采样**：眼睛保持注视画面中闪烁的黄色目标点，按下 `空格键` 开始连续稳定采样，直到完成全部 9 个采样点。
4. 标定参数将自动更新至 `config/mapping.json`。

---

### 4. 启动实时追踪与控制

确保 ESP32 已通电，启动上位机视觉控制主程序：

```bash
# 默认模式 (自动搜索连接 ESP32 并使用默认摄像头)
python scripts/run.py

# 纯视觉算法调试模式 (不连接蓝牙硬件)
python scripts/run.py --no-ble

# 使用笔记本自带摄像头 (Index: 0)
python scripts/run.py --builtin-cam
```

- **状态说明**：
  - `TRACKING` (绿色)：实时稳定跟踪眼球。
  - `PREDICTING` (黄色)：短暂眨眼或反光丢失，卡尔曼根据物理速度外推补全。
  - `LOST (CENTERED)` (红色)：超过最大容忍丢失帧数，仿生眼平滑回中（$X=0, Y=0$）。
  - 按键盘 `Q` 键或 `空格键` 均可安全退出。

---

## 工具链与高级使用

### 模型训练与格式转换

项目提供了一套从样本采集、清洗到训练部署的全套工具链：

```bash
# 1. 采集自定义样本 (按空格拍照，按 Q 保存退出)
python tools/capture_samples.py

# 2. 清洗标注并排查 train 与 test 之间的数据泄露
python tools/prepare_dataset.py --data-root datasets/eye_dataset --keep-classes 0

# 3. 划分数据集 (默认 8:1:1)
python tools/split_dataset.py

# 4. 启动高精度 YOLOv8 训练
python tools/train.py --config config/train_config.json

# 5. 导出 ONNX 与 TensorRT FP16 引擎并进行输出精度校验
python tools/export.py --weights runs/detect/runs/train/eye_detector_v2-2/weights/best.pt --imgsz 640

# 6. 多模型基准延迟与精度评测
python tools/benchmark.py --models models/best.pt models/best.engine --images datasets/eye_dataset/test/images
```

---

### 墨水屏点阵素材转换

若需要替换眼白、瞳孔、眼睑皮肤的样式图，请将图片放置于 `images/` 目录下后执行：

```bash
python images/convert.py
```
脚本将自动执行：
- 图像居中裁剪与缩放至 $200 \times 200$。
- **Floyd-Steinberg** 误差扩散单色抖动。
- 提取眼眶 Alpha 通道开孔掩码（1-bit Mask）。
- 自动生成合规的 C 头文件：`yanbai.h`、`tongkong.h`、`yankuang.h`。
- 输出多角度静态合成预览图及全流程动态模拟 `eye_turn_simulation.gif`。

---

### BLE 手动调试

使用内置命令行工具直接向已连接的 ESP32 发送任意移动指令：

```bash
python scripts/test_ble_esp32.py
```
在控制台中输入目标坐标（例如 `15 8` 或 `-10 0`）验证屏幕物理显示范围。

---

## 配置说明

核心配置文件位于 `config/settings.json`：

```json
{
    "model": {
        "engine_path": "models/best.engine",    // TensorRT 模型路径
        "cpu_fallback_path": "models/best.pt",  // CPU 备用 PyTorch 权重
        "device": "auto",                       // "auto", "cpu", 0
        "image_size": 640,
        "confidence": 0.40,
        "iou": 0.40
    },
    "camera": {
        "index": 0,                             // 摄像头设备编号
        "width": 640,
        "height": 480
    },
    "mapping": {
        "file": "config/mapping.json",          // 标定多项式参数文件
        "min_x": -35, "max_x": 35,              // X 轴物理位移边界
        "min_y": -18, "max_y": 18               // Y 轴物理位移边界
    },
    "filter": {
        "ewma_weights": [0.2, 0.3, 0.5],        // 平滑滤波权重
        "max_prediction_frames": 25,            // 最大断测帧数 (约 0.8s)
        "auto_center_when_lost": true           // 目标丢失后是否自动归零
    },
    "ble": {
        "device_name": "ESP32_EPD_EYE",
        "service_uuid": "4fafc201-1fb5-459e-8fcc-c5c9c331914b",
        "characteristic_uuid": "beb5483e-36e1-4688-b7f5-ea07361b26a8",
        "minimum_interval": 0.05,               // 最小发送间隔 (50ms)
        "keepalive_interval": 0.35              // 心跳保活间隔
    }
}
```

---

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。