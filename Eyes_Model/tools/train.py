# -*- coding: utf-8 -*-
"""YOLOv8 高精度眼部目标检测训练程序。"""

from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="训练高精度单类别眼部检测模型")
    parser.add_argument("--config", type=Path, default=Path("config/train_config.json"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = 0 if (cfg["device"] == "auto" and torch.cuda.is_available()) else cfg["device"]
    yaml_path = Path(cfg["dataset_yaml"]).resolve()

    print(f"[*] 启动训练，计算设备: {device} | 骨干模型: {cfg['base_model']} | 数据配置: {yaml_path}")

    model = YOLO(cfg["base_model"])
    project_dir = Path("runs/train")

    train_results = model.train(
        data=str(yaml_path),
        epochs=cfg.get("epochs", 100),
        imgsz=cfg.get("image_size", 640),
        batch=cfg.get("batch_size", 16),
        patience=cfg.get("patience", 25),
        workers=cfg.get("workers", 4),
        device=device,
        seed=cfg.get("seed", 42),
        project=str(project_dir),
        name="eye_detector_v2",
        exist_ok=False,
        amp=(str(device) != "cpu"),
        rect=cfg.get("rect", True),                # 启用矩形训练加速与防畸变
        close_mosaic=cfg.get("close_mosaic", 10),  # 训练结束前 10 轮关闭 mosaic，提升细长/精细边缘回归精度
        flipud=0.0,                                # 眼睛绝不会发生上下颠倒，禁用翻转增强
        fliplr=0.5,                                # 允许左右镜像增强
        save=True,
        save_period=10,
        verbose=True,
    )

    best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"训练完成但未找到最佳权重: {best_pt}")

    print(f"[+] 最优模型已生成: {best_pt}，正在执行全量验证集指标评估...")
    val_model = YOLO(str(best_pt))
    metrics = val_model.val(data=str(yaml_path), device=device, imgsz=cfg["image_size"], rect=True)

    metadata = {
        "model_sha256": compute_file_sha256(best_pt),
        "export_time": datetime.datetime.now().isoformat(),
        "input_image_size": cfg["image_size"],
        "base_architecture": cfg["base_model"],
        "classes": ["eye"],
        "evaluation_metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision_mean": float(metrics.box.mp),
            "recall_mean": float(metrics.box.mr),
        },
    }

    meta_file = best_pt.parent / "model_metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] 训练元数据已写入: {meta_file}")
    print(f"    >>> 最终 mAP@0.5: {metrics.box.map50:.4f} | mAP@0.5:0.95: {metrics.box.map:.4f} <<<")

if __name__ == "__main__":
    main()