# -*- coding: utf-8 -*-
"""实时视觉追踪管线：集成非阻塞取流、YOLO 推理、多目标仲裁与目标丢失回中控制。"""

from __future__ import annotations
import argparse
import asyncio
import logging
import math
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .config import DEFAULT_CONFIG, load_settings, project_path
from .core import EyeMapper, KalmanBoxFilter, WeightedSmoother
from .hardware import BLETransmitter

LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "Bionic Eye Tracker (Host Control)"


class ThreadedCamera:
    """专用后台守护线程采集相机帧，防止 capture.read() 阻塞 asyncio 事件循环。"""

    def __init__(self, index: int, width: int, height: int):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开索引为 {index} 的摄像头设备")

        self.ret = False
        self.frame: np.ndarray | None = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        while self.running:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


def select_device(model_path: Path, configured: str | int) -> str | int:
    """自适应选择计算后端。"""
    if configured != "auto":
        return configured
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    if model_path.suffix.lower() == ".engine":
        raise RuntimeError("TensorRT 引擎要求 CUDA GPU 环境，请使用 .pt 模型进行 CPU 推理")
    return "cpu"


def select_model(settings: dict[str, Any]) -> Path:
    """根据硬件条件选择模型引擎或备用权重。"""
    m_set = settings["model"]
    engine = project_path(settings, m_set["engine_path"])
    fallback = project_path(settings, m_set["cpu_fallback_path"])

    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    target = engine if use_cuda and engine.is_file() else fallback
    if not target.is_file():
        raise FileNotFoundError(f"未找到可用模型权重文件: {target}")
    return target


def best_detection(boxes: Any) -> tuple[float, float, float, float, float] | None:
    """从 YOLO 多目标结果中提取最高置信度边框与置信度值。"""
    if boxes is None or len(boxes) == 0:
        return None
    idx = int(boxes.conf.argmax().item()) if boxes.conf is not None else 0
    conf = float(boxes.conf[idx].item()) if boxes.conf is not None else 0.0
    x1, y1, x2, y2 = boxes.xyxy[idx].detach().cpu().numpy()
    return float(x1), float(y1), float(x2), float(y2), conf


def draw_box(frame: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int], label: str = "") -> None:
    """在画板上绘制标注边框与标签。"""
    cx, cy, w, h = box
    pt1 = (int(cx - w / 2), int(cy - h / 2))
    pt2 = (int(cx + w / 2), int(cy + h / 2))
    cv2.rectangle(frame, pt1, pt2, color, 2)
    if label:
        cv2.putText(frame, label, (pt1[0], max(15, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


async def track(settings: dict[str, Any], transmitter: BLETransmitter | None) -> None:
    model_path = select_model(settings)
    device = select_device(model_path, settings["model"]["device"])
    LOGGER.info(f"加载 YOLO 权重: {model_path.name} @ 设备: {device}")
    model = YOLO(str(model_path))

    mapping = settings["mapping"]
    mapper = EyeMapper(
        project_path(settings, mapping["file"]),
        (mapping["min_x"], mapping["max_x"], mapping["min_y"], mapping["max_y"]),
    )
    smoother = WeightedSmoother(settings["filter"]["ewma_weights"])
    kalman = KalmanBoxFilter()
    max_pred_frames = int(settings["filter"]["max_prediction_frames"])
    auto_center = bool(settings["filter"].get("auto_center_when_lost", True))

    cam_cfg = settings["camera"]
    camera = ThreadedCamera(int(cam_cfg["index"]), int(cam_cfg["width"]), int(cam_cfg["height"]))

    lost_frames = 0
    sent_x, sent_y = 0, 0
    pupil_px_x, pupil_px_y = 0.0, 0.0
    det_conf = 0.0
    status_str = "SEARCHING"
    status_color = (0, 165, 255)

    # === 【新增】门限保护参数与计数器 ===
    MAX_JUMP_DIST = 90.0        # 单帧允许的最大跳变像素（近眼特写通常眼球转动单帧不会超过 60~90 像素）
    jump_reject_frames = 0     # 连续异常跳变帧数计数器

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                await asyncio.sleep(0.005)
                continue

            # YOLO 目标检测
            res = model.predict(
                frame,
                imgsz=int(settings["model"]["image_size"]),
                conf=float(settings["model"]["confidence"]),
                iou=float(settings["model"]["iou"]),
                verbose=False,
                device=device,
            )[0]

            detection_res = best_detection(res.boxes)
            filtered: tuple[float, float, float, float] | None = None

            if detection_res is not None:
                x1, y1, x2, y2, det_conf = detection_res
                pupil_px_x = (x1 + x2) / 2.0
                pupil_px_y = (y1 + y2) / 2.0
                measured = (pupil_px_x, pupil_px_y, x2 - x1, y2 - y1)
                
                # 绘制原始检测红框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

                # ==================== 位移合理性门限 (Gating Check) ====================
                is_outlier = False
                if kalman.initialized:
                    last_kx, last_ky = kalman.box[0], kalman.box[1]
                    jump_dist = math.hypot(pupil_px_x - last_kx, pupil_px_y - last_ky)
                    
                    # 瞬间位移超出合理生理极限，判定为反光/阴影误检
                    if jump_dist > MAX_JUMP_DIST:
                        jump_reject_frames += 1
                        # 连续异常少于 6 帧时，拒收此测量值
                        if jump_reject_frames < 6:
                            is_outlier = True
                        else:
                            # 连续多次出现新位置，说明是头部/摄像头物理移动，强制接受并重置
                            LOGGER.info("目标发生物理长位移，强制重新锁定")
                            kalman.initialize(*measured)
                            jump_reject_frames = 0
                    else:
                        jump_reject_frames = 0
                # =====================================================================

                if is_outlier:
                    # 遭遇孤立误检噪点：不把错误坐标喂给卡尔曼，使用物理模型预测外推
                    filtered = kalman.predict()
                    status_str = f"GLITCH REJECTED ({jump_dist:.0f}px)"
                    status_color = (0, 255, 255)
                    draw_box(frame, filtered, (0, 255, 255), "FILTERED NOISE")
                else:
                    if not kalman.initialized:
                        kalman.initialize(*measured)
                    else:
                        kalman.predict()
                        kalman.update(*measured)

                    filtered = kalman.box
                    lost_frames = 0
                    status_str = "TRACKING"
                    status_color = (0, 255, 0)
                    draw_box(frame, filtered, (0, 255, 0), f"TRACKED {det_conf:.2f}")

            elif kalman.initialized and lost_frames < max_pred_frames:
                # 眨眼/短暂反光丢失：通过物理速度外推
                filtered = kalman.predict()
                lost_frames += 1
                status_str = f"PREDICTING ({lost_frames}/{max_pred_frames})"
                status_color = (0, 255, 255)
                draw_box(frame, filtered, (0, 255, 255), f"LOST:{lost_frames}")

            elif kalman.initialized:
                # 超过容忍极限：平滑回中重置
                LOGGER.info("目标长时间丢失，眼球自动回归居中位置 (0, 0)")
                kalman.reset()
                smoother.reset()
                status_str = "LOST (CENTERED)"
                status_color = (0, 0, 255)
                sent_x, sent_y = 0, 0
                if auto_center and transmitter:
                    transmitter.publish(0, 0)

            # 映射并发布坐标
            if filtered is not None:
                sm_x, sm_y = smoother.update(filtered[0], filtered[1])
                sent_x, sent_y = mapper.map(sm_x, sm_y)

                if transmitter:
                    transmitter.publish(sent_x, sent_y)

            # ========================= 调试 UI 叠加 =========================
            # 1. 正在发送给下位机的目标物理坐标
            cv2.putText(
                frame,
                f"SENT TO BLE: X={sent_x}, Y={sent_y}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            # 2. 状态与置信度
            cv2.putText(
                frame,
                f"STATUS: {status_str} | CONF: {det_conf:.2f}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # 3. 画面像素坐标与物理限位
            cv2.putText(
                frame,
                f"Pupil Pixel: ({pupil_px_x:.1f}, {pupil_px_y:.1f}) | Range: X[{mapping['min_x']},{mapping['max_x']}] Y[{mapping['min_y']},{mapping['max_y']}]",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            # 4. 退出按键提示
            cv2.putText(
                frame,
                "Press [SPACE] or [q] to exit",
                (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # =================== 退出方式：支持空格键 (32) 与 q 键 ===================
            key = cv2.waitKey(1) & 0xFF
            if key in (32, ord("q"), ord("Q")):  # 32 为空格键 ASCII
                LOGGER.info("检测到按键退出指令 (SPACE/Q)")
                break

            await asyncio.sleep(0.001)

    finally:
        camera.release()
        cv2.destroyAllWindows()


async def run(config_path: Path, no_ble: bool, builtin_cam: bool = False) -> None:
    settings = load_settings(config_path)

    if builtin_cam:
        settings["camera"]["index"] = 0
        LOGGER.info("已切换为电脑自带摄像头 (Index: 0)")
    else:
        LOGGER.info(f"使用默认摄像头 (Index: {settings['camera']['index']})")

    transmitter = None if no_ble else BLETransmitter(settings["ble"])
    send_task = None

    if transmitter:
        send_task = asyncio.create_task(transmitter.run(), name="BLETransmitterTask")

    try:
        await track(settings, transmitter)
    finally:
        if transmitter:
            await transmitter.stop()
        if send_task:
            send_task.cancel()
            with suppress(asyncio.CancelledError):
                await send_task


def main() -> None:
    parser = argparse.ArgumentParser(description="仿生机器人眼上位机控制系统")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument("--no-ble", action="store_true", help="纯视觉调试模式，不开启 BLE 连接")
    parser.add_argument("--builtin-cam", action="store_true", help="使用电脑自带摄像头（默认使用外接 USB 摄像头）")
    parser.add_argument("--log-level", default="INFO", help="日志输出级别")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # if sys.platform.startswith("win"):
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(run(args.config, args.no_ble, args.builtin_cam))
    except KeyboardInterrupt:
        LOGGER.info("接收到用户退出中断信号")


if __name__ == "__main__":
    main()