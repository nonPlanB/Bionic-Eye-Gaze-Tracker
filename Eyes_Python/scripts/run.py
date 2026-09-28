"""上位机启动主脚本入口。"""

import sys
from pathlib import Path

# 将 src 目录挂载进系统模块检索链
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artificial_eye.tracker import main

if __name__ == "__main__":
    main()