import cv2
from pathlib import Path

# 保存目录
save_dir = Path("datasets/raw_images")
save_dir.mkdir(parents=True, exist_ok=True)

# 0 代表电脑自带摄像头，1 代表外接 USB 摄像头
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = len(list(save_dir.glob("*.jpg")))
print("=== 图像采集已启动 ===")
print("1. 对准眼睛，转动视线")
print("2. 按【空格键】拍摄并保存一张图像")
print("3. 按【Q 键】或【ESC】退出\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    cv2.putText(display, f"Captured: {count} | Press SPACE to capture, Q to quit", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Eye Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # 空格键
        count += 1
        img_path = save_dir / f"eye_{count:04d}.jpg"
        cv2.imwrite(str(img_path), frame)
        print(f"[+] 已保存: {img_path}")
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()