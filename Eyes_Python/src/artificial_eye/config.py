# -*- coding: utf-8 -*-
"""全局路径管理与 JSON 配置加载器。"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

# 项目根目录自动定位
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.json"


def load_settings(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取并解析 settings.json，注入根目录元数据。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)

    settings["_config_path"] = str(config_path)
    settings["_project_root"] = str(PROJECT_ROOT)
    return settings


def project_path(settings: dict[str, Any], relative_or_absolute: str) -> Path:
    """将配置中的相对路径转为基于项目根目录的绝对路径。"""
    p = Path(relative_or_absolute).expanduser()
    if not p.is_absolute():
        p = Path(settings["_project_root"]) / p
    return p.resolve()