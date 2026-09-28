"""ESP32-Arduino BLE 链路手动交互调试工具。"""

import argparse
import asyncio
from bleak import BleakClient, BleakScanner

async def run_test(device_name: str, char_uuid: str):
    print(f"[*] 正在搜索 Arduino 设备: {device_name} ...")
    device = await BleakScanner.find_device_by_name(device_name, timeout=6.0)
    if not device:
        print(f"[!] 未找到名称为 '{device_name}' 的设备，请确保 ESP32 已经上电并运行 Eyes.ino")
        return

    print(f"[+] 找到设备: {device.name} [{device.address}]，正在建立 GATT 连接...")
    async with BleakClient(device) as client:
        print("[+] 连接成功！")
        print("[i] 请输入目标偏移坐标测试眼球运动，格式: X Y (例如: 10 5, -15 0, 0 0)")
        print("[i] 输入 q 退出调试\n")

        while True:
            raw = await asyncio.to_thread(input, "EyeCmd > ")
            raw = raw.strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            try:
                parts = raw.split()
                if len(parts) != 2:
                    print("[!] 格式错误，请输入以空格隔开的两个整数")
                    continue
                x, y = int(parts[0]), int(parts[1])
            except ValueError:
                print("[!] 输入非法字符，必须是整数")
                continue

            # 组装符合 parse_input_coordinates 解析器的 ASCII 格式
            payload = f"{x},{y}".encode("ascii")
            await client.write_gatt_char(char_uuid, payload, response=False)
            print(f"[>] 发送成功: {payload.decode()} (约束区间: X[-20,20], Y[-12,12])")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="ESP32_EPD_EYE", help="与 Arduino Eyes.ino 对应的广播名")
    parser.add_argument("--char", default="beb5483e-36e1-4688-b7f5-ea07361b26a8", help="特征值 UUID")
    args = parser.parse_args()

    asyncio.run(run_test(args.device, args.char))

if __name__ == "__main__":
    main()