import random
import shutil
from pathlib import Path

# 配置源数据路径（根据上一步导出目录设置）
# 若导出时勾选了 Save with images，图片和 txt 都在 datasets/labels
SOURCE_DIR = Path("datasets/labels")
# 如果图片仍在 datasets/raw_images，脚本会自动兼容读取
RAW_IMG_DIR = Path("datasets/raw_images")

TARGET_ROOT = Path("datasets/eye_dataset")
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, valid, test
SEED = 42

random.seed(SEED)

# 1. 查找所有有效的 txt 标注文件并配对图片
pairs = []
txt_files = list(SOURCE_DIR.glob("*.txt"))

for txt_path in txt_files:
    if txt_path.name == "classes.txt":
        continue
    
    stem = txt_path.stem
    # 优先在 SOURCE_DIR 找同名图片，找不到再去 RAW_IMG_DIR 找
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = SOURCE_DIR / f"{stem}{ext}"
        if candidate.is_file():
            img_path = candidate
            break
        candidate_raw = RAW_IMG_DIR / f"{stem}{ext}"
        if candidate_raw.is_file():
            img_path = candidate_raw
            break

    if img_path and img_path.is_file():
        pairs.append((img_path, txt_path))

total = len(pairs)
print(f"[*] 成功配对到有效样本: {total} 组")
if total == 0:
    print("[!] 错误: 未找到匹配的 (图片 + txt) 数据，请检查路径。")
    exit(1)

# 2. 随机打散
random.shuffle(pairs)

# 3. 计算分割点 (161, 20, 20)
n_train = int(total * SPLIT_RATIO[0])
n_val = int(total * SPLIT_RATIO[1])

splits = {
    "train": pairs[:n_train],
    "valid": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:]
}

# 4. 复制文件到项目标准目录
for split_name, items in splits.items():
    img_dest = TARGET_ROOT / split_name / "images"
    lbl_dest = TARGET_ROOT / split_name / "labels"
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    for img_p, txt_p in items:
        shutil.copy2(img_p, img_dest / img_p.name)
        shutil.copy2(txt_p, lbl_dest / txt_p.name)

    print(f"[+] [{split_name.upper()}] 划分完成: {len(items)} 组样本")

print(f"\n[✓] 数据集已成功构建在: {TARGET_ROOT.resolve()}")