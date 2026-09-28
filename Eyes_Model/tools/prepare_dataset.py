# -*- coding: utf-8 -*-
"""生产级数据集清洗工具：支持按源类别白名单过滤、单类别对齐 (class 0) 与跨集泄露检测。"""

from __future__ import annotations
import argparse
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def sanitize_and_filter_labels(label_path: Path, target_class_ids: set[str] | None = None) -> int:
    """按白名单保留目标类别，并将类别 ID 统一对齐为 0 (eye)。"""
    if not label_path.is_file():
        return 0

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sanitized_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cls_id = parts[0]
            # 如果指定了目标类别白名单，不在白名单中的框直接剔除（例如排除人脸框 face）
            if target_class_ids is not None and cls_id not in target_class_ids:
                continue

            # 统一对齐为单类别 0
            parts[0] = "0"
            sanitized_lines.append(" ".join(parts))

    if sanitized_lines:
        label_path.write_text("\n".join(sanitized_lines) + "\n", encoding="utf-8")
    else:
        # 如果过滤后无有效标注，清空文件
        label_path.write_text("", encoding="utf-8")

    return len(sanitized_lines)

def audit_and_clean_dataset(dataset_root: Path, target_class_ids: list[str] | None = None) -> None:
    print(f"[*] 开始检查并清洗数据集: {dataset_root}")
    target_set = set(target_class_ids) if target_class_ids else None
    if target_set:
        print(f"[*] 启用类别白名单过滤，仅提取类别 ID: {target_set}")

    splits = ["train", "valid", "test"]
    split_stems: dict[str, set[str]] = {}

    for split in splits:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            print(f"[!] 警告: 目录不存在 {img_dir}")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        stems = {img.stem.split(".rf.")[0] for img in images}
        split_stems[split] = stems

        box_count = 0
        empty_labels = 0
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                cnt = sanitize_and_filter_labels(lbl_file, target_set)
                box_count += cnt
                if cnt == 0:
                    empty_labels += 1

        print(f"[{split.upper()}] 有效图像数: {len(images)}, 保留有效眼部框数: {box_count} (空标签数: {empty_labels})")

    if "train" in split_stems and "test" in split_stems:
        overlap = split_stems["train"].intersection(split_stems["test"])
        if overlap:
            print(f"[!] 警告: 检测到数据泄露！共有 {len(overlap)} 个源样本在 train 与 test 中重复。")
        else:
            print("[+] 数据泄露检查通过: 训练集与测试集完全独立。")

def main():
    parser = argparse.ArgumentParser(description="YOLO 眼部数据集清洗与标准化工具")
    parser.add_argument("--data-root", type=Path, required=True, help="数据集根目录路径")
    parser.add_argument("--keep-classes", nargs="*", default=None, help="原始标注中属于眼睛的类别 ID，如: --keep-classes 0 1")
    args = parser.parse_args()

    audit_and_clean_dataset(args.data_root.resolve(), args.keep_classes)

if __name__ == "__main__":
    main()