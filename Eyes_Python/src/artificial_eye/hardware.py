"""BLE 异步通信传输器，负责与 Arduino（Eyes.ino）维持低时延无线控制。"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class BLETransmitter:
    """仅保留最新坐标、抗抖动并支持丢包重连的 BLE 发射端。"""

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.client = None
        self._latest: tuple[int, int] | None = None
        self._last_sent: tuple[int, int] | None = None
        self._last_send_time = 0.0
        self._changed = asyncio.Event()
        self._stopping = False

    def publish(self, x: int, y: int) -> None:
        """发布最新的目标注视坐标 (x, y)。"""
        state = (int(x), int(y))
        if state != self._latest:
            self._latest = state
            self._changed.set()

    async def _connect(self) -> bool:
        from bleak import BleakClient, BleakScanner

        target_name = self.settings["device_name"]
        target_uuid = self.settings["service_uuid"]
        LOGGER.info(f"正在扫描 BLE 设备: {target_name} (UUID: {target_uuid})...")

        # 优先通过名字精确扫描，若无则尝试通过广播服务 UUID 扫描
        device = await BleakScanner.find_device_by_name(
            target_name, timeout=self.settings["scan_timeout"]
        )
        if device is None:
            LOGGER.warning(f"未能通过名称找到设备，尝试通过 Service UUID 检索...")
            devices = await BleakScanner.discover(timeout=self.settings["scan_timeout"])
            for d in devices:
                uuids = [u.lower() for u in d.metadata.get("uuids", [])]
                if target_uuid.lower() in uuids:
                    device = d
                    break

        if device is None:
            LOGGER.warning("未找到匹配的 ESP32 墨水屏设备")
            return False

        try:
            self.client = BleakClient(device, timeout=10.0)
            await self.client.connect()
            LOGGER.info(f"BLE 成功连接至: {device.name or target_name} [{device.address}]")
            return bool(self.client.is_connected)
        except Exception as err:
            LOGGER.error(f"建立 BLE 连接异常: {err}")
            self.client = None
            return False

    async def run(self) -> None:
        """主协程循环：处理坐标差分发送与定时心跳维持。"""
        while not self._stopping:
            try:
                if self.client is None or not self.client.is_connected:
                    if not await self._connect():
                        await asyncio.sleep(self.settings["reconnect_interval"])
                        continue

                # 等待坐标更新，或者达到保活心跳阈值
                keepalive = float(self.settings["keepalive_interval"])
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=keepalive)
                except asyncio.TimeoutError:
                    pass

                self._changed.clear()
                if self._latest is None:
                    continue

                # 限制发送频率，防止上位机洪水挤占 BLE 栈
                elapsed = time.monotonic() - self._last_send_time
                min_interval = float(self.settings["minimum_interval"])
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

                # 生成严格匹配 Eyes.ino parse_input_coordinates 的格式: "X,Y"
                payload = f"{self._latest[0]},{self._latest[1]}".encode("ascii")
                
                await self.client.write_gatt_char(
                    self.settings["characteristic_uuid"],
                    payload,
                    response=False  # Write Without Response，实现最高刷新率
                )
                self._last_sent = self._latest
                self._last_send_time = time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.warning(f"BLE 通信链路异常断开，准备重连: {e}")
                await self._disconnect()
                await asyncio.sleep(self.settings["reconnect_interval"])

    async def _disconnect(self) -> None:
        if self.client is not None:
            try:
                if self.client.is_connected:
                    await self.client.disconnect()
            except Exception:
                pass
        self.client = None

    async def stop(self) -> None:
        self._stopping = True
        self._changed.set()
        await self._disconnect()