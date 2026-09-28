# -*- coding: utf-8 -*-
"""生产级眼部目标检测推理引擎：支持 TensorRT Engine、ONNX 与 PyTorch 动态后备与预热。"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import numpy as np
from ultralytics import YOLO


class EyeDetector:
    """眼部目标检测引擎，封装硬件自适应推理、启动预热与置信度裁决。"""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.60,
        iou_threshold: float = 0.40,
        image_size: int = 640,
        device: str | int = "auto",
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到指定的检测模型文件: {self.model_path}")

        self.conf = conf_threshold
        self.iou = iou_threshold
        self.image_size = image_size
        self.device = self._resolve_device(device)

        # 加载模型并执行 GPU 预热
        self.model = YOLO(str(self.model_path))
        self._warmup()

    def _resolve_device(self, device_config: str | int) -> str | int:
        if device_config != "auto":
            return device_config
        try:
            import torch
            if torch.cuda.is_available():
                return 0
        except ImportError:
            pass
        if self.model_path.suffix.lower() == ".engine":
            raise RuntimeError("TensorRT .engine 必须运行在带 CUDA 的 NVIDIA 设备上")
        return "cpu"

    def _warmup(self) -> None:
        """执行预热推理，防止首帧加载权重引起延迟突刺。"""
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.image_size, conf=0.01, device=self.device, verbose=False)

    def detect(self, frame: np.ndarray) -> tuple[float, float, float, float, float] | None:
        """输入 BGR 图像帧，返回最优目标的 (中心点x, 中心点y, 宽w, 高h, 置信度conf)。

        未检出目标时返回 None。
        """
        if frame is None or frame.size == 0:
            return None

        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        # 选取置信度最高的目标框（解决多目标仲裁问题）
        best_idx = int(boxes.conf.argmax().item()) if boxes.conf is not None else 0
        conf_score = float(boxes.conf[best_idx].item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].detach().cpu().numpy()

        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        w = float(x2 - x1)
        h = float(y2 - y1)

        return cx, cy, w, h, conf_score

    def profile_latency(self, frame: np.ndarray, iterations: int = 50) -> dict[str, float]:
        """对当前输入做多轮前向推理，返回平均预处理、推理与后处理延迟 (ms)。"""
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _ = self.detect(frame)
            times.append((time.perf_counter_ns() - t0) / 1_000_000.0)

        return {
            "mean_latency_ms": float(np.mean(times)),
            "median_latency_ms": float(np.median(times)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "theoretical_fps": float(1000.0 / np.mean(times)),
        }