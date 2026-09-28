"""数学变换与运动滤波算法核心库：包含多项式坐标映射、加权平滑与快速响应卡尔曼滤波。"""

from __future__ import annotations
import json
from collections import deque
from pathlib import Path
from typing import Iterable
import numpy as np


class EyeMapper:
    """非线性二次多项式视线空间映射器。"""

    def __init__(self, mapping_file: Path, bounds: tuple[int, int, int, int]):
        if not mapping_file.is_file():
            raise FileNotFoundError(f"映射参数文件缺失: {mapping_file}")

        with mapping_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.x_coeffs = self._validate(data.get("X_COEFFS"), "X_COEFFS")
        self.y_coeffs = self._validate(data.get("Y_COEFFS"), "Y_COEFFS")
        self.min_x, self.max_x, self.min_y, self.max_y = bounds

    @staticmethod
    def _validate(values: object, name: str) -> np.ndarray:
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError(f"{name} 必须包含恰好 5 个多项式系数")
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError(f"{name} 中包含非有限数值(NaN/Inf)")
        return result

    def map(self, cx: float, cy: float) -> tuple[int, int]:
        """依据 cx, cy 像素坐标，计算符合 Arduino 物理限位的偏移量。"""
        # 特征向量: [cx^2, cy^2, cx, cy, 1]
        terms = np.array([cx * cx, cy * cy, cx, cy, 1.0], dtype=float)
        raw_x = float(self.x_coeffs @ terms)
        raw_y = float(self.y_coeffs @ terms)

        # 【修改这里】：对 raw_x 取负，反转左右极性
        raw_x = -raw_x

        # 严格约束到 Arduino 的物理开孔区间
        clamped_x = int(round(np.clip(raw_x, self.min_x, self.max_x)))
        clamped_y = int(round(np.clip(raw_y, self.min_y, self.max_y)))
        return clamped_x, clamped_y


class WeightedSmoother:
    """多帧指数滑动加权平滑器（消除边缘毛刺）。"""

    def __init__(self, weights: Iterable[float]):
        self.weights = np.asarray(list(weights), dtype=float)
        if len(self.weights) == 0 or np.any(self.weights <= 0):
            raise ValueError("EWMA 权重序列必须全为正数")
        self.x_values: deque[float] = deque(maxlen=len(self.weights))
        self.y_values: deque[float] = deque(maxlen=len(self.weights))

    def reset(self) -> None:
        self.x_values.clear()
        self.y_values.clear()

    def update(self, x: float, y: float) -> tuple[float, float]:
        self.x_values.append(float(x))
        self.y_values.append(float(y))

        # 归一化局部窗口权重
        w = self.weights[-len(self.x_values):]
        w = w / w.sum()
        return (
            float(np.dot(np.asarray(self.x_values), w)),
            float(np.dot(np.asarray(self.y_values), w)),
        )


class KalmanBoxFilter:
    """针对人眼快速跳视（Saccade）优化的 8 状态恒速卡尔曼滤波器。"""

    def __init__(self, dt: float = 1.0):
        # 状态向量: [x, y, vx, vy, w, h, vw, vh]^T
        self.state = np.zeros((8, 1), dtype=float)
        self.transition = np.eye(8, dtype=float)
        for p, v in ((0, 2), (1, 3), (4, 6), (5, 7)):
            self.transition[p, v] = dt

        # 观测矩阵：仅直接观测中心与尺寸 [x, y, w, h]
        self.observation = np.zeros((4, 8), dtype=float)
        self.observation[0, 0] = self.observation[1, 1] = 1.0
        self.observation[2, 4] = self.observation[3, 5] = 1.0

        self.covariance = np.eye(8, dtype=float) * 50.0
        
        # 提高位置与速度的过程噪声方差（由 0.001 调至 0.25），提高对人眼高速转动的跟踪敏捷性
        self.process_noise = np.eye(8, dtype=float) * 0.25
        # 适度降低测量噪声（由 80.0 调至 15.0），增强对高置信度 YOLO 框的跟随意图
        self.measurement_noise = np.eye(4, dtype=float) * 15.0
        self.initialized = False

    def reset(self) -> None:
        self.state.fill(0)
        self.covariance = np.eye(8, dtype=float) * 50.0
        self.initialized = False

    def initialize(self, cx: float, cy: float, width: float, height: float) -> None:
        self.state[:, 0] = (cx, cy, 0.0, 0.0, width, height, 0.0, 0.0)
        self.initialized = True

    def predict(self) -> tuple[float, float, float, float]:
        if not self.initialized:
            raise RuntimeError("卡尔曼滤波器尚未初始化，不可执行预测")
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.box

    def update(self, cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
        measurement = np.array([[cx], [cy], [width], [height]], dtype=float)
        residual = self.observation @ self.covariance @ self.observation.T + self.measurement_noise
        gain = self.covariance @ self.observation.T @ np.linalg.pinv(residual)
        
        # 状态更新
        self.state += gain @ (measurement - self.observation @ self.state)

        # 约瑟夫稳定形式协方差更新: P = (I - KH)P(I - KH)^T + KRK^T
        i_kh = np.eye(8) - gain @ self.observation
        self.covariance = i_kh @ self.covariance @ i_kh.T + gain @ self.measurement_noise @ gain.T
        return self.box

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[4, 0]),
            float(self.state[5, 0]),
        )