from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
WIDTH = 200
HEIGHT = 200

# 素材与导出映射表
TASKS = [
    {
        "source": "yanbai.png",
        "array_name": "gImage_sclera",
        "header_guard": "_YANBAI_H_",
        "header_out": "yanbai.h",
        "extract_mask": False,
    },
    {
        "source": "tongkong.png",
        "array_name": "gImage_pupil",
        "header_guard": "_TONGKONG_H_",
        "header_out": "tongkong.h",
        "extract_mask": False,
    },
    {
        "source": "yankuang.png",
        "array_name": "gImage_eyelid",
        "mask_array_name": "gImage_eyelid_mask",
        "header_guard": "_YANKUANG_H_",
        "header_out": "yankuang.h",
        "extract_mask": True,
    },
]


def prepare_layer(src_path: Path, extract_mask: bool = False):
    """转换单图层：输出灰度图、1-bit 抖动图及开孔遮罩"""
    original = Image.open(src_path)

    # 1. 提取眼眶开孔的二值遮罩 (255=皮肤区域, 0=透明眼孔)
    mask_mono = None
    if extract_mask:
        if original.mode in ("RGBA", "LA") or "transparency" in original.info:
            alpha = original.convert("RGBA").split()[-1]
            alpha_square = ImageOps.fit(
                alpha, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
            # Alpha > 128 判定为眼睑覆盖区，小于 128 判定为开孔
            mask_mono = alpha_square.point(lambda p: 255 if p > 128 else 0, mode="1")
        else:
            # 若无透明通道，将中心区域纯白视为开孔，其余为眼眶
            rgb = original.convert("RGB")
            sq = ImageOps.fit(rgb, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
            mask_mono = sq.convert("L").point(lambda p: 0 if p > 240 else 255, mode="1")

    # 2. RGB 纹理转换为 1-bit 墨水屏位图
    if original.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", original.size, (0, 0, 0))
        bg.paste(original, mask=original.split()[-1])
        source_rgb = bg
    else:
        source_rgb = original.convert("RGB")

    square = ImageOps.fit(
        source_rgb, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )

    gray = ImageOps.grayscale(square)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.2)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.1, percent=140, threshold=3))

    # Floyd-Steinberg 误差扩散抖动 (1=白, 0=黑)
    mono = gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG)

    return gray, mono, mask_mono


def pack_vendor_frame(mono: Image.Image) -> bytes:
    """按行打包为 5000 字节点阵数组 (MSB first, 1=白, 0=黑)"""
    packed = bytearray()
    for y in range(HEIGHT):
        for x0 in range(0, WIDTH, 8):
            val = 0
            for bit in range(8):
                if mono.getpixel((x0 + bit, y)) != 0:
                    val |= (0x80 >> bit)
            packed.append(val)
    return bytes(packed)


def write_c_header(file_path: Path, guard: str, arrays: list[tuple[str, bytes]]):
    """写入 C 语言头文件"""
    lines = [f"#ifndef {guard}", f"#define {guard}", ""]
    for name, data in arrays:
        lines.append(f"// 200x200, 1 bit/pixel, row-major, MSB first; 1=white, 0=black.")
        lines.append(f"const unsigned char {name}[5000] = {{")
        for offset in range(0, len(data), 16):
            chunk = data[offset : offset + 16]
            suffix = "," if offset + 16 < len(data) else ""
            lines.append("  " + ",".join(f"0X{b:02X}" for b in chunk) + suffix)
        lines.append("};\n")
    lines.append(f"#endif // {guard}\n")
    file_path.write_text("\n".join(lines), encoding="ascii")


def simulate_composite(sclera_mono: Image.Image, pupil_mono: Image.Image, 
                       eyelid_mono: Image.Image, eyelid_mask: Image.Image, 
                       shift_x: int, shift_y: int) -> Image.Image:
    """模拟单片机合成三层图像：底层眼白 -> 中层移动瞳孔 -> 顶层眼眶遮罩"""
    canvas = sclera_mono.copy()
    center_x, center_y, radius = 100, 100, 44
    r2 = radius * radius

    # 叠加中层瞳孔
    for dy in range(-radius, radius + 1):
        dy2 = dy * dy
        for dx in range(-radius, radius + 1):
            if dx * dx + dy2 <= r2:
                src_x = center_x + dx
                src_y = center_y + dy
                dst_x = src_x + shift_x
                dst_y = src_y + shift_y
                if 0 <= dst_x < WIDTH and 0 <= dst_y < HEIGHT:
                    canvas.putpixel((dst_x, dst_y), pupil_mono.getpixel((src_x, src_y)))

    # 叠加顶层眼眶
    if eyelid_mono and eyelid_mask:
        for y in range(HEIGHT):
            for x in range(WIDTH):
                # 遮罩为 255 表示眼眶皮肤，覆盖底层
                if eyelid_mask.getpixel((x, y)) != 0:
                    canvas.putpixel((x, y), eyelid_mono.getpixel((x, y)))

    return canvas


def main():
    processed = {}

    # 1. 批量处理并保存各单图层图片
    for task in TASKS:
        src = ROOT / task["source"]
        if not src.exists():
            print(f"[错误] 未找到文件: {src.name}")
            return

        print(f"-> 正在转换: {src.name} ...")
        gray, mono, mask = prepare_layer(src, task["extract_mask"])
        stem = src.stem

        # 保存 1-bit 单色图与灰度图
        mono.save(ROOT / f"{stem}_mono_1bit.png")
        gray.save(ROOT / f"{stem}_gray.png")
        if mask:
            mask.save(ROOT / f"{stem}_mask_1bit.png")

        processed[stem] = {"mono": mono, "mask": mask}

        # 写入对应 .h 头文件
        arrays_to_write = [(task["array_name"], pack_vendor_frame(mono))]
        if mask:
            arrays_to_write.append((task["mask_array_name"], pack_vendor_frame(mask)))
        write_c_header(ROOT / task["header_out"], task["header_guard"], arrays_to_write)

    print("\n-> 正在合成三图层预览效果图与动态模拟图...")

    sclera = processed["yanbai"]["mono"]
    pupil = processed["tongkong"]["mono"]
    eyelid = processed["yankuang"]["mono"]
    eyelid_mask = processed["yankuang"]["mask"]

    # 2. 生成多角度静态合成效果图
    simulate_composite(sclera, pupil, eyelid, eyelid_mask, 0, 0).save(ROOT / "eye_composite_center.png")
    simulate_composite(sclera, pupil, eyelid, eyelid_mask, -18, 0).save(ROOT / "eye_composite_look_left.png")
    simulate_composite(sclera, pupil, eyelid, eyelid_mask, 18, 0).save(ROOT / "eye_composite_look_right.png")
    simulate_composite(sclera, pupil, eyelid, eyelid_mask, 0, -12).save(ROOT / "eye_composite_look_up.png")
    simulate_composite(sclera, pupil, eyelid, eyelid_mask, 0, 12).save(ROOT / "eye_composite_look_down.png")

    # 3. 生成眼球连续转动的仿真动图 (GIF)
    gif_frames = []
    # 模拟眼球转动轨迹坐标序列 (shift_x, shift_y)
    motion_path = [
        (0, 0), (8, 0), (16, 0), (22, 0), (16, 5), (8, 10), 
        (0, 12), (-8, 10), (-16, 5), (-22, 0), (-16, -5), 
        (-8, -10), (0, -12), (8, -10), (16, -5), (0, 0)
    ]
    for sx, sy in motion_path:
        frame = simulate_composite(sclera, pupil, eyelid, eyelid_mask, sx, sy)
        # 转为 L 模式以支持流畅 GIF 编码
        gif_frames.append(frame.convert("L"))

    gif_frames[0].save(
        ROOT / "eye_turn_simulation.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=180,
        loop=0
    )

    print("转换与渲染完成！生成的所有图片及头文件已保存至当前目录。")


if __name__ == "__main__":
    main()