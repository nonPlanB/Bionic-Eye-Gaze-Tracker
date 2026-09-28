# -*- coding: utf-8 -*-
"""可复现的模型基准评测工具：对比 .pt、.onnx、.engine 在固定测试集上的速度、IoU与中心点误差。"""

from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def ground_truth_box(image_path: Path, width: int, height: int) -> np.ndarray:
    """读取对应的 YOLO 标签文件中的 ground truth 真实框坐标。"""
    label_path = image_path.parents[1] / "labels" / f"{image_path.stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"未找到标签文件: {label_path}")
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{label_path} 中无标注目标")
    _, cx, cy, w, h = map(float, lines[0].split())
    return np.array([
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    ])


def evaluate_model(model_path: Path, images_dir: Path, imgsz: int, threshold: float, device: str) -> dict:
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"目录中未找到图像: {images_dir}")

    model = YOLO(str(model_path))
    # 预热单张
    model.predict(str(images[0]), imgsz=imgsz, conf=0.01, device=device, verbose=False)

    latencies = []
    center_errors = []
    hits = 0

    for img_p in images:
        frame = cv2.imread(str(img_p))
        h, w = frame.shape[:2]
        gt = ground_truth_box(img_p, w, h)

        t0 = time.perf_counter_ns()
        res = model.predict(frame, imgsz=imgsz, conf=0.01, device=device, verbose=False)[0]
        latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue

        best_idx = int(boxes.conf.argmax().item())
        conf = float(boxes.conf[best_idx].item())
        pred = boxes.xyxy[best_idx].detach().cpu().numpy()

        pred_center = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
        gt_center = np.array([(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2])
        err = float(np.linalg.norm(pred_center - gt_center))
        center_errors.append(err)

        # 判定 Hit (置信度达标且中心误差小于 15 像素)
        if conf >= threshold and err <= 15.0:
            hits += 1

    return {
        "model": model_path.name,
        "format": model_path.suffix,
        "device": device,
        "image_count": len(images),
        "hit_rate": hits / len(images),
        "center_error_px_mean": statistics.fmean(center_errors) if center_errors else None,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / statistics.fmean(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="多模型性能与精度横向基准评测")
    parser.add_argument("--models", nargs="+", type=Path, required=True, help="待评测模型列表 (.pt/.engine/.onnx)")
    parser.add_argument("--images", type=Path, required=True, help="测试集图像文件夹")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    results = []
    for m in args.models:
        dev = "cpu" if m.suffix == ".pt" and args.device == "cpu" else args.device
        res = evaluate_model(m.resolve(), args.images.resolve(), args.imgsz, args.threshold, dev)
        results.append(res)
        print(f"[{res['model']}] FPS: {res['fps']:.1f} | 均值延迟: {res['latency_mean_ms']:.2f} ms | 命中率: {res['hit_rate']*100:.1f}%")

    print("\n完整评测结果 JSON:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()