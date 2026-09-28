"""九点视线空间标定程序 (点击对角自动生成 9 点 + 动态间距调节版)"""

from __future__ import annotations
import argparse
import json
import math
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 将 src 目录挂载进系统搜索路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_eye.config import DEFAULT_CONFIG, load_settings, project_path
from artificial_eye.tracker import best_detection, select_device, select_model

# 墨水屏 9 点物理目标位移 (固定映射关系: 上中下 x 左中右)
TARGET_DISPLACEMENTS = [
    # 上排
    (-18, -10), (0, -10), (18, -10),
    # 中排
    (-18,   0), (0,   0), (18,   0),
    # 下排
    (-18,  10), (0,  10), (18,  10),
]

POINT_NAMES = [
    "Top-Left", "Top-Center", "Top-Right",
    "Mid-Left", "Center",     "Mid-Right",
    "Btm-Left", "Btm-Center", "Btm-Right",
]


class StabilityGatedSampler:
    """注视稳定性门限采样器：仅在注视点连续稳定处于设定半径内才累加样本。"""

    def __init__(self, stable_frames: int = 8, stability_radius: float = 4.0):
        self.stable_frames = stable_frames
        self.stability_radius = stability_radius
        self._recent: deque[tuple[float, float]] = deque(maxlen=stable_frames)
        self.samples: list[tuple[float, float]] = []
        self.status = "stabilizing"

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def reset(self) -> None:
        self._recent.clear()
        self.samples.clear()
        self.status = "stabilizing"

    def update(self, pupil: tuple[float, float] | None) -> bool:
        if pupil is None:
            self._recent.clear()
            self.samples.clear()
            self.status = "no_detection"
            return False

        point = (float(pupil[0]), float(pupil[1]))
        if self._recent:
            center_x = sum(p[0] for p in self._recent) / len(self._recent)
            center_y = sum(p[1] for p in self._recent) / len(self._recent)
            if math.hypot(point[0] - center_x, point[1] - center_y) > self.stability_radius:
                self._recent.clear()
                self.samples.clear()
                self._recent.append(point)
                self.status = "unstable"
                return False

        self._recent.append(point)
        if len(self._recent) < self.stable_frames:
            self.status = "stabilizing"
            return False

        self.samples.append(point)
        self.status = "collecting"
        return True


def beep(frequency: int, duration: int) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()
    except Exception:
        pass


def fit_polynomial(raw_points: list[tuple[float, float]], target_points: list[tuple[int, int]]) -> tuple[list[float], list[float]]:
    raw = np.asarray(raw_points, dtype=float)
    target = np.asarray(target_points, dtype=float)

    matrix = np.column_stack([
        raw[:, 0] ** 2,
        raw[:, 1] ** 2,
        raw[:, 0],
        raw[:, 1],
        np.ones(len(raw))
    ])

    x_coeffs, *_ = np.linalg.lstsq(matrix, target[:, 0], rcond=None)
    y_coeffs, *_ = np.linalg.lstsq(matrix, target[:, 1], rcond=None)
    return x_coeffs.tolist(), y_coeffs.tolist()


def generate_9_points(pt1: tuple[int, int], pt2: tuple[int, int], spacing_scale: float = 1.0) -> list[list[int]]:
    """根据两个对角点和缩放参数生成 3x3 均匀分布的 9 个标定点。"""
    x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    half_w = ((x_max - x_min) / 2.0) * spacing_scale
    half_h = ((y_max - y_min) / 2.0) * spacing_scale

    xs = [int(round(cx - half_w)), int(round(cx)), int(round(cx + half_w))]
    ys = [int(round(cy - half_h)), int(round(cy)), int(round(cy + half_h))]

    points = []
    idx = 0
    for y in ys:
        for x in xs:
            tgt_x, tgt_y = TARGET_DISPLACEMENTS[idx]
            points.append([x, y, tgt_x, tgt_y])
            idx += 1
    return points


def main():
    parser = argparse.ArgumentParser(description="对角点框选与间距可调的 9 点注视空间标定程序")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="全局配置文件路径")
    parser.add_argument("--samples", type=int, default=30, help="每个标定点收集的有效样本帧数")
    parser.add_argument("--spacing", type=float, default=1.0, help="9 点间距默认缩放系数 (例如 0.8 为收缩20%%, 1.2 为扩散20%%)")
    args = parser.parse_args()

    settings = load_settings(args.config)
    model_path = select_model(settings)
    device = select_device(model_path, settings["model"]["device"])
    
    print(f"[*] 载入检测模型: {model_path.name} @ 设备: {device}")
    model = YOLO(str(model_path))

    cam_cfg = settings["camera"]
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(int(cam_cfg["index"]), backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头 (Index: {cam_cfg['index']})")

    win_name = "Bionic Eye Calibration (Diagonal 9-Points Mode)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)

    # 阶段状态机:
    # "DIAG_P1": 等待点击第 1 个对角点
    # "DIAG_P2": 等待点击第 2 个对角点
    # "GRID_READY": 9 点已生成，允许按 +/- 微调间距，按 SPACE 确认开始标定
    # "CALIBRATING": 正在执行 9 点顺序凝视采样
    stage = "DIAG_P1"

    diag_p1: tuple[int, int] | None = None
    diag_p2: tuple[int, int] | None = None
    spacing_scale = max(0.2, float(args.spacing))
    points: list[list[int]] = []

    calib_idx = 0
    calib_state = "waiting"  # "waiting" | "active"
    sampler = StabilityGatedSampler(stable_frames=8, stability_radius=4.0)

    # 鼠标事件回调
    def on_mouse(event, x, y, flags, param):
        nonlocal stage, diag_p1, diag_p2, points, spacing_scale
        if event == cv2.EVENT_LBUTTONDOWN:
            if stage == "DIAG_P1":
                diag_p1 = (int(x), int(y))
                stage = "DIAG_P2"
                beep(900, 80)
            elif stage == "DIAG_P2":
                diag_p2 = (int(x), int(y))
                # 检查两个对角点不能过近
                if abs(diag_p2[0] - diag_p1[0]) > 30 and abs(diag_p2[1] - diag_p1[1]) > 30:
                    points = generate_9_points(diag_p1, diag_p2, spacing_scale)
                    stage = "GRID_READY"
                    beep(1200, 100)
                else:
                    print("[!] 对角间距过小，请重新选择对角点 2")

    cv2.setMouseCallback(win_name, on_mouse)

    raw_points: list[tuple[float, float]] = []
    target_points: list[tuple[int, int]] = []

    print("\n================== 对角 9 点标定操作指南 ==================")
    print("【步骤 1】在画面中用【鼠标左键】依次点击两个对角（如舒适区域的左上、右下）。")
    print("【步骤 2】系统自动生成 9 点矩形阵列：")
    print("         - 按键盘【+】或【=】：扩大 9 点分布间距")
    print("         - 按键盘【-】或【_】：缩小 9 点分布间距")
    print("         - 按键盘【R】：重新点击对角点")
    print("         - 按【空格键】：锁定网格并正式开始注视标定")
    print("【步骤 3】标定时注视黄色圆点，按【空格键】启动当前点采样。按 'Q' 可随时退出。\n")

    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            res = model.predict(
                frame,
                imgsz=int(settings["model"]["image_size"]),
                conf=0.55,
                verbose=False,
                device=device
            )[0]

            det = best_detection(res.boxes)
            pupil = None
            if det:
                x1, y1, x2, y2, _conf = det
                pupil = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.drawMarker(frame, (int(pupil[0]), int(pupil[1])), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)

            # ---------------- 阶段 1: 点击第 1 个对角 ----------------
            if stage == "DIAG_P1":
                cv2.putText(frame, "Step 1: Click Diagonal Corner 1 (e.g. Top-Left)", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # ---------------- 阶段 2: 点击第 2 个对角 ----------------
            elif stage == "DIAG_P2":
                cv2.drawMarker(frame, diag_p1, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.putText(frame, "Step 2: Click Diagonal Corner 2 (e.g. Bottom-Right)", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # ---------------- 阶段 3: 9 点网格预览与间距微调 ----------------
            elif stage == "GRID_READY":
                # 绘制外框与 9 点预览
                x_min, x_max = min(diag_p1[0], diag_p2[0]), max(diag_p1[0], diag_p2[0])
                y_min, y_max = min(diag_p1[1], diag_p2[1]), max(diag_p1[1], diag_p2[1])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1)

                for idx, p in enumerate(points):
                    cv2.circle(frame, (p[0], p[1]), 8, (0, 255, 255), -1)
                    cv2.putText(frame, str(idx + 1), (p[0] + 10, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                tip = f"Spacing: {spacing_scale:.2f}x | [+]/[-] Adjust | [R] Reset | [SPACE] Start"
                cv2.putText(frame, tip, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ---------------- 阶段 4: 正式开始 9 点注视采样 ----------------
            elif stage == "CALIBRATING":
                # 绘制历史已采样的点
                for done_idx in range(calib_idx):
                    hx, hy = points[done_idx][0], points[done_idx][1]
                    cv2.circle(frame, (hx, hy), 6, (120, 120, 120), -1)
                    cv2.putText(frame, str(done_idx + 1), (hx + 8, hy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

                pt_x, pt_y, tgt_x, tgt_y = points[calib_idx]
                dot_color = (0, 255, 255)
                name_en = POINT_NAMES[calib_idx]
                tip = f"[{calib_idx + 1}/9 {name_en}] Look at dot, Press [SPACE]"

                if calib_state == "active":
                    sampler.update(pupil)
                    if sampler.status == "no_detection":
                        dot_color = (0, 0, 255)
                        tip = "No pupil detected! Keep looking."
                    elif sampler.status in ("unstable", "stabilizing"):
                        dot_color = (0, 165, 255)
                        tip = "Hold your gaze still..."
                    elif sampler.status == "collecting":
                        dot_color = (0, 255, 0)
                        tip = f"Sampling: {sampler.sample_count}/{args.samples}"

                    if sampler.sample_count >= args.samples:
                        median_pt = np.median(np.asarray(sampler.samples), axis=0)
                        raw_points.append((float(median_pt[0]), float(median_pt[1])))
                        target_points.append((tgt_x, tgt_y))

                        beep(1600, 200)
                        calib_idx += 1
                        calib_state = "waiting"

                        if calib_idx >= len(points):
                            # 全部标定结束
                            break
                        continue

                # 绘制当前注视目标点
                cv2.circle(frame, (pt_x, pt_y), 14, dot_color, -1)
                cv2.circle(frame, (pt_x, pt_y), 22, dot_color, 2)
                cv2.drawMarker(frame, (pt_x, pt_y), (0, 0, 0), cv2.MARKER_CROSS, 14, 1)
                cv2.putText(frame, tip, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if calib_state == "active" else (0, 255, 255), 2)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # 按键逻辑响应
            if key in (ord("q"), ord("Q"), 27):
                print("[!] 取消标定，未更改配置。")
                return

            if stage == "GRID_READY":
                # 按 + 或 = 扩大间距
                if key in (ord("+"), ord("=")):
                    spacing_scale = min(1.8, spacing_scale + 0.05)
                    points = generate_9_points(diag_p1, diag_p2, spacing_scale)
                # 按 - 或 _ 缩小间距
                elif key in (ord("-"), ord("_")):
                    spacing_scale = max(0.3, spacing_scale - 0.05)
                    points = generate_9_points(diag_p1, diag_p2, spacing_scale)
                # 按 R 重新选对角
                elif key in (ord("r"), ord("R")):
                    stage = "DIAG_P1"
                    diag_p1, diag_p2 = None, None
                    points.clear()
                # 按空格确认网格并启动标定
                elif key == 32:
                    stage = "CALIBRATING"
                    calib_idx = 0
                    calib_state = "waiting"
                    beep(800, 150)

            elif stage == "CALIBRATING":
                if key == 32 and calib_state == "waiting":
                    calib_state = "active"
                    sampler.reset()
                    beep(800, 120)

    finally:
        capture.release()
        cv2.destroyAllWindows()

    # 拟合多项式并保存
    x_coeffs, y_coeffs = fit_polynomial(raw_points, target_points)
    dest_file = project_path(settings, settings["mapping"]["file"])
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = dest_file.with_suffix(".json.tmp")
    tmp_file.write_text(
        json.dumps({"X_COEFFS": x_coeffs, "Y_COEFFS": y_coeffs}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp_file.replace(dest_file)

    beep(2000, 350)
    print(f"\n[✓] 九点视线校准成功！参数已写入: {dest_file}")


if __name__ == "__main__":
    main()