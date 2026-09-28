This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
images/
  convert.py
  eye_composite_center.png
  eye_composite_look_down.png
  eye_composite_look_left.png
  eye_composite_look_right.png
  eye_composite_look_up.png
  eye_turn_simulation.gif
  FullEye.png
  tongkong_gray.png
  tongkong_mono_1bit.png
  tongkong.png
  yanbai_gray.png
  yanbai_mono_1bit.png
  yanbai.png
  yankuang_gray.png
  yankuang_mask_1bit.png
  yankuang_mono_1bit.png
  yankuang.png
Display_EPD_W21_spi.cpp
Display_EPD_W21_spi.h
Display_EPD_W21.cpp
Display_EPD_W21.h
Eyes.ino
tongkong.h
yanbai.h
yankuang.h
```

# Files

## File: images/convert.py
```python
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
```

## File: Display_EPD_W21_spi.cpp
```cpp
#include "Display_EPD_W21_spi.h"
#include <SPI.h>

//SPI write byte
void SPI_Write(unsigned char value)
{				   			 
   SPI.transfer(value);
}

//SPI write command
void EPD_W21_WriteCMD(unsigned char command)
{
	EPD_W21_CS_0;
	EPD_W21_DC_0;  // D/C#   0:command  1:data  
	SPI_Write(command);
	EPD_W21_CS_1;
}
//SPI write data
void EPD_W21_WriteDATA(unsigned char datas)
{
	EPD_W21_CS_0;
	EPD_W21_DC_1;  // D/C#   0:command  1:data
	SPI_Write(datas);
	EPD_W21_CS_1;
}
```

## File: Display_EPD_W21_spi.h
```c
#ifndef _DISPLAY_EPD_W21_SPI_
#define _DISPLAY_EPD_W21_SPI_
#include "Arduino.h"

//IO settings
// ESP32-M1 schematic mapping:
// SCLK=GPIO18, MOSI=GPIO23, BUSY=GPIO13, RES=GPIO12,
// D/C=GPIO14, CS=GPIO27.
#define isEPD_W21_BUSY digitalRead(13)  // BUSY / schematic A14
#define EPD_W21_RST_0 digitalWrite(12,LOW)  // RES / schematic A15
#define EPD_W21_RST_1 digitalWrite(12,HIGH)
#define EPD_W21_DC_0  digitalWrite(14,LOW) // D/C / schematic A16
#define EPD_W21_DC_1  digitalWrite(14,HIGH)
#define EPD_W21_CS_0 digitalWrite(27,LOW) // CS / schematic A17
#define EPD_W21_CS_1 digitalWrite(27,HIGH)


void SPI_Write(unsigned char value);
void EPD_W21_WriteDATA(unsigned char datas);
void EPD_W21_WriteCMD(unsigned char command);


#endif
```

## File: Display_EPD_W21.cpp
```cpp
#include "Display_EPD_W21_spi.h"
#include "Display_EPD_W21.h"

//Delay Functions
void delay_xms(unsigned int xms)
{
	delay(xms);//At least 10ms delay 
}
////////////////////////////////////E-paper demo//////////////////////////////////////////////////////////
//Busy function
void Epaper_READBUSY(void)
{ 
  while(1)
  {	 //=1 BUSY
     if(isEPD_W21_BUSY==0) break;
  }  
}
//Full screen refresh initialization
void EPD_HW_Init(void)
{
	EPD_W21_RST_0;  // Module reset   
	delay_xms(10);//At least 10ms delay 
	EPD_W21_RST_1;
	delay_xms(10); //At least 10ms delay 
	
	Epaper_READBUSY();   
	EPD_W21_WriteCMD(0x12);  //SWRESET
	Epaper_READBUSY();   
		
	EPD_W21_WriteCMD(0x01); //Driver output control      
	EPD_W21_WriteDATA((EPD_HEIGHT-1)%256);    
	EPD_W21_WriteDATA((EPD_HEIGHT-1)/256);
	EPD_W21_WriteDATA(0x00);

	EPD_W21_WriteCMD(0x11); //data entry mode       
	EPD_W21_WriteDATA(0x01);

	EPD_W21_WriteCMD(0x44); //set Ram-X address start/end position   
	EPD_W21_WriteDATA(0x00);
	EPD_W21_WriteDATA(EPD_WIDTH/8-1);    

	EPD_W21_WriteCMD(0x45); //set Ram-Y address start/end position          
	EPD_W21_WriteDATA((EPD_HEIGHT-1)%256);    
	EPD_W21_WriteDATA((EPD_HEIGHT-1)/256);
	EPD_W21_WriteDATA(0x00);
	EPD_W21_WriteDATA(0x00); 

	EPD_W21_WriteCMD(0x3C); //BorderWavefrom
	EPD_W21_WriteDATA(0x05);	
	  	
  EPD_W21_WriteCMD(0x18); //Read built-in temperature sensor
	EPD_W21_WriteDATA(0x80);	

	EPD_W21_WriteCMD(0x4E);   // set RAM x address count to 0;
	EPD_W21_WriteDATA(0x00);
	EPD_W21_WriteCMD(0x4F);   // set RAM y address count to 0X199;    
	EPD_W21_WriteDATA((EPD_HEIGHT-1)%256);    
	EPD_W21_WriteDATA((EPD_HEIGHT-1)/256);
  Epaper_READBUSY();
	
}


//////////////////////////////Display Update Function///////////////////////////////////////////////////////
//Full screen refresh update function
void EPD_Update(void)
{   
  EPD_W21_WriteCMD(0x22); //Display Update Control
  EPD_W21_WriteDATA(0xF7);   
  EPD_W21_WriteCMD(0x20); //Activate Display Update Sequence
  Epaper_READBUSY();   

}

//Partial refresh update function
void EPD_Part_Update(void)
{
	EPD_W21_WriteCMD(0x22); //Display Update Control
	EPD_W21_WriteDATA(0xFF);   
	EPD_W21_WriteCMD(0x20); //Activate Display Update Sequence
	Epaper_READBUSY(); 			
}
//////////////////////////////Display Data Transfer Function////////////////////////////////////////////
//Full screen refresh display function
void EPD_WhiteScreen_ALL(const unsigned char *datas)
{
   unsigned int i;	
  EPD_W21_WriteCMD(0x24);   //write RAM for black(0)/white (1)
  for(i=0;i<EPD_ARRAY;i++)
   {               
     EPD_W21_WriteDATA(datas[i]);
   }
   EPD_Update();	 
}

//Clear screen display
void EPD_WhiteScreen_White(void)
{
 unsigned int i;
 EPD_W21_WriteCMD(0x24);   //write RAM for black(0)/white (1)
 for(i=0;i<EPD_ARRAY;i++)
 {
		EPD_W21_WriteDATA(0xff);
	}
	EPD_Update();
}
//Display all black
void EPD_WhiteScreen_Black(void)
{
 unsigned int i;
 EPD_W21_WriteCMD(0x24);   //write RAM for black(0)/white (1)
 for(i=0;i<EPD_ARRAY;i++)
 {
		EPD_W21_WriteDATA(0x00);
	}
	EPD_Update();
}
//Partial refresh of background display, this function is necessary, please do not delete it!!!
void EPD_SetRAMValue_BaseMap( const unsigned char * datas)
{
	unsigned int i;   	
  EPD_W21_WriteCMD(0x24);   //Write Black and White image to RAM
  for(i=0;i<EPD_ARRAY;i++)
   {               
     EPD_W21_WriteDATA(datas[i]);
   }
  EPD_W21_WriteCMD(0x26);   //Write Black and White image to RAM
  for(i=0;i<EPD_ARRAY;i++)
   {               
     EPD_W21_WriteDATA(datas[i]);
   }
   EPD_Update();		 
	 
}
//Partial refresh display
void EPD_Dis_Part(unsigned int x_start,unsigned int y_start,const unsigned char * datas,unsigned int PART_COLUMN,unsigned int PART_LINE)
{
	unsigned int i;  
	unsigned int x_end,y_end;
	
	x_start=x_start/8; //x address start
	x_end=x_start+PART_LINE/8-1; //x address end
	y_start=y_start; //Y address start
	y_end=y_start+PART_COLUMN-1; //Y address end
	
	EPD_W21_RST_0;  // Module reset   
	delay_xms(10);//At least 10ms delay 
	EPD_W21_RST_1;
	delay_xms(10); //At least 10ms delay 	
	EPD_W21_WriteCMD(0x3C); //BorderWavefrom,
	EPD_W21_WriteDATA(0x80);	
	
	EPD_W21_WriteCMD(0x44);       // set RAM x address start/end
	EPD_W21_WriteDATA(x_start);  //x address start
	EPD_W21_WriteDATA(x_end);   //y address end   
	EPD_W21_WriteCMD(0x45);    // set RAM y address start/end
	EPD_W21_WriteDATA(y_start%256);  //y address start2 
	EPD_W21_WriteDATA(y_start/256); //y address start1 
	EPD_W21_WriteDATA(y_end%256);  //y address end2 
	EPD_W21_WriteDATA(y_end/256); //y address end1   

	EPD_W21_WriteCMD(0x4E);        // set RAM x address count to 0;
	EPD_W21_WriteDATA(x_start);   //x start address
	EPD_W21_WriteCMD(0x4F);      // set RAM y address count to 0X127;    
	EPD_W21_WriteDATA(y_start%256);//y address start2
	EPD_W21_WriteDATA(y_start/256);//y address start1
	
	
	 EPD_W21_WriteCMD(0x24);   //Write Black and White image to RAM
   for(i=0;i<PART_COLUMN*PART_LINE/8;i++)
   {                         
     EPD_W21_WriteDATA(datas[i]);
   } 
	 EPD_Part_Update();

}
//Full screen partial refresh display
void EPD_Dis_PartAll(const unsigned char * datas)
{
	unsigned int i;  
	unsigned int PART_COLUMN, PART_LINE;
	PART_COLUMN=EPD_HEIGHT,PART_LINE=EPD_WIDTH;

	EPD_W21_RST_0;  // Module reset   
	delay_xms(10); //At least 10ms delay 
	EPD_W21_RST_1;
	delay_xms(10); //At least 10ms delay 	
	EPD_W21_WriteCMD(0x3C); //BorderWavefrom,
	EPD_W21_WriteDATA(0x80);	


	EPD_W21_WriteCMD(0x24);   //Write Black and White image to RAM
	 for(i=0;i<PART_COLUMN*PART_LINE/8;i++)
	 {                         
		 EPD_W21_WriteDATA(datas[i]);
	 } 
	 EPD_Part_Update();

}
//Deep sleep function   
void EPD_DeepSleep(void)
{  	
  EPD_W21_WriteCMD(0x10); //Enter deep sleep
  EPD_W21_WriteDATA(0x01); 
  delay_xms(100);
}

//Partial refresh write address and data
void EPD_Dis_Part_RAM(unsigned int x_start,unsigned int y_start,const unsigned char * datas,unsigned int PART_COLUMN,unsigned int PART_LINE)
{
	unsigned int i;  
	unsigned int x_end,y_end;
	
	x_start=x_start/8; //x address start
	x_end=x_start+PART_LINE/8-1; //x address end
	
	y_start=y_start-1; //Y address start
	y_end=y_start+PART_COLUMN-1; //Y address end
	
	EPD_W21_RST_0;  // Module reset   
	delay_xms(10);//At least 10ms delay 
	EPD_W21_RST_1;
	delay_xms(10); //At least 10ms delay 	
	EPD_W21_WriteCMD(0x3C); //BorderWavefrom,
	EPD_W21_WriteDATA(0x80);		
	
	EPD_W21_WriteCMD(0x44);       // set RAM x address start/end
	EPD_W21_WriteDATA(x_start);  //x address start
	EPD_W21_WriteDATA(x_end);   //y address end   
	EPD_W21_WriteCMD(0x45);     // set RAM y address start/end
	EPD_W21_WriteDATA(y_start%256);  //y address start2 
	EPD_W21_WriteDATA(y_start/256); //y address start1 
	EPD_W21_WriteDATA(y_end%256);  //y address end2 
	EPD_W21_WriteDATA(y_end/256); //y address end1   

	EPD_W21_WriteCMD(0x4E);   // set RAM x address count to 0;
	EPD_W21_WriteDATA(x_start);   //x start address
	EPD_W21_WriteCMD(0x4F);   // set RAM y address count to 0X127;    
	EPD_W21_WriteDATA(y_start%256); //y address start2
	EPD_W21_WriteDATA(y_start/256); //y address start1
		
	EPD_W21_WriteCMD(0x24);   //Write Black and White image to RAM
  for(i=0;i<PART_COLUMN*PART_LINE/8;i++)
   {                         
     EPD_W21_WriteDATA(datas[i]);
   } 
}
//Clock display
void EPD_Dis_Part_Time(unsigned int x_startA,unsigned int y_startA,const unsigned char * datasA,
	                       unsigned int x_startB,unsigned int y_startB,const unsigned char * datasB,
												 unsigned int x_startC,unsigned int y_startC,const unsigned char * datasC,
												 unsigned int x_startD,unsigned int y_startD,const unsigned char * datasD,
											   unsigned int x_startE,unsigned int y_startE,const unsigned char * datasE,
												 unsigned int PART_COLUMN,unsigned int PART_LINE
	                      )
{
	EPD_Dis_Part_RAM(x_startA,y_startA,datasA,PART_COLUMN,PART_LINE);
	EPD_Dis_Part_RAM(x_startB,y_startB,datasB,PART_COLUMN,PART_LINE);
	EPD_Dis_Part_RAM(x_startC,y_startC,datasC,PART_COLUMN,PART_LINE);
	EPD_Dis_Part_RAM(x_startD,y_startD,datasD,PART_COLUMN,PART_LINE);
	EPD_Dis_Part_RAM(x_startE,y_startE,datasE,PART_COLUMN,PART_LINE);
	EPD_Part_Update();
}												 




////////////////////////////////Other newly added functions////////////////////////////////////////////
//Display rotation 180 degrees initialization
void EPD_HW_Init_180(void)
{
	EPD_W21_RST_0;  // Module reset   
	delay_xms(10); //At least 10ms delay 
	EPD_W21_RST_1;
	delay_xms(10); //At least 10ms delay 
	
	Epaper_READBUSY();   
	EPD_W21_WriteCMD(0x12);  //SWRESET
	Epaper_READBUSY();   
	
	EPD_W21_WriteCMD(0x3C); //BorderWavefrom
	EPD_W21_WriteDATA(0x05);
	
	EPD_W21_WriteCMD(0x01); //Driver output control      
	EPD_W21_WriteDATA((EPD_HEIGHT-1)%256);    
	EPD_W21_WriteDATA((EPD_HEIGHT-1)/256);
	EPD_W21_WriteDATA(0x00); 

	EPD_W21_WriteCMD(0x11); //data entry mode       
	EPD_W21_WriteDATA(0x02);

	EPD_W21_WriteCMD(0x44); //set Ram-X address start/end position   
	EPD_W21_WriteDATA(EPD_WIDTH/8-1);    
	EPD_W21_WriteDATA(0x00);  

	EPD_W21_WriteCMD(0x45); //set Ram-Y address start/end position          
	EPD_W21_WriteDATA(0x00);
	EPD_W21_WriteDATA(0x00); 
  EPD_W21_WriteDATA((EPD_HEIGHT-1)%256);  
	EPD_W21_WriteDATA((EPD_HEIGHT-1)/256);

	
  EPD_W21_WriteCMD(0x18); //Read built-in temperature sensor
	EPD_W21_WriteDATA(0x80);	

	EPD_W21_WriteCMD(0x4E);   // set RAM x address count to 0;
	EPD_W21_WriteDATA(EPD_WIDTH/8-1);  
	EPD_W21_WriteCMD(0x4F);   // set RAM y address count to 0X199;    
	EPD_W21_WriteDATA(0x00);
	EPD_W21_WriteDATA(0x00);
  Epaper_READBUSY();
}


/***********************************************************
						end file
***********************************************************/
```

## File: Display_EPD_W21.h
```c
#ifndef _DISPLAY_EPD_W21_H_
#define _DISPLAY_EPD_W21_H_


#define EPD_WIDTH   200  
#define EPD_HEIGHT  200
#define EPD_ARRAY  EPD_WIDTH*EPD_HEIGHT/8  


//Full screen refresh display
void EPD_HW_Init(void); 
void EPD_HW_Init_180(void);	
void EPD_WhiteScreen_ALL(const unsigned char *datas);
void EPD_WhiteScreen_White(void);
void EPD_WhiteScreen_Black(void);
void EPD_DeepSleep(void);
//Partial refresh display 
void EPD_SetRAMValue_BaseMap(const unsigned char * datas);
void EPD_Dis_PartAll(const unsigned char * datas);
void EPD_Dis_Part(unsigned int x_start,unsigned int y_start,const unsigned char * datas,unsigned int PART_COLUMN,unsigned int PART_LINE);
void EPD_Dis_Part_Time(unsigned int x_startA,unsigned int y_startA,const unsigned char * datasA,
	                       unsigned int x_startB,unsigned int y_startB,const unsigned char * datasB,
												 unsigned int x_startC,unsigned int y_startC,const unsigned char * datasC,
												 unsigned int x_startD,unsigned int y_startD,const unsigned char * datasD,
											   unsigned int x_startE,unsigned int y_startE,const unsigned char * datasE,
												 unsigned int PART_COLUMN,unsigned int PART_LINE
	                      );												 
  
												 
#endif
```

## File: Eyes.ino
```
#include <SPI.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

// 墨水屏底层驱动
#include "Display_EPD_W21_spi.h"
#include "Display_EPD_W21.h"

// 引入三图层素材头文件
#include "yanbai.h"     // 底层：眼白 (gImage_sclera)
#include "tongkong.h"   // 中层：瞳孔 (gImage_pupil)
#include "yankuang.h"   // 顶层：眼眶与开孔掩码 (gImage_eyelid, gImage_eyelid_mask)

// --------------------------- 图层开关 ---------------------------
#define HAS_SCLERA_LAYER 1  // 底层眼白开关 (1=开)
#define HAS_EYELID_LAYER 1  // 顶层眼眶与遮罩开关 (1=开)

// --------------------------- 硬件引脚定义 ---------------------------
constexpr uint8_t EPD_SCK  = 18;
constexpr uint8_t EPD_MOSI = 23;
constexpr uint8_t EPD_CS   = 27;
constexpr uint8_t EPD_DC   = 14;
constexpr uint8_t EPD_RST  = 12;
constexpr uint8_t EPD_BUSY = 13;

// --------------------------- 画布与眼球几何参数 ---------------------------
#define EPD_W 200
#define EPD_H 200
#define BUFFER_SIZE (EPD_W * EPD_H / 8) // 5000 字节

// 瞳孔几何参数 (仅裁剪提取瞳孔内部圆形区域，外围黑边透明化)
constexpr int PUPIL_SRC_CENTER_X = 100;
constexpr int PUPIL_SRC_CENTER_Y = 100;
constexpr int PUPIL_RADIUS       = 44;   // 瞳孔半径

// 根据扁长型眼眶实际开孔大小，限制瞳孔的物理位移极限
constexpr int MAX_OFFSET_X       = 35;   // X 轴最大偏移量 (左右看)
constexpr int MAX_OFFSET_Y       = 18;   // Y 轴最大偏移量 (上下看)

static uint8_t epd_canvas[BUFFER_SIZE];

// 坐标与刷新节流管理
int posX = 0;
int posY = 0;
int lastX = 999;
int lastY = 999;
bool newPositionReceived = false;

uint32_t lastRefreshTime = 0;
const uint32_t MIN_REFRESH_INTERVAL_MS = 350; // 局部刷新节流保护（至少 350ms）
uint16_t partialRefreshCounter = 0;

// --------------------------- 像素操作函数 ---------------------------
// 读取位图指定坐标像素：1 为白，0 为黑
static inline uint8_t get_bitmap_pixel(const uint8_t *bitmap, int x, int y) {
    if (x < 0 || x >= EPD_W || y < 0 || y >= EPD_H) return 0;
    uint8_t b = bitmap[y * (EPD_W / 8) + (x / 8)];
    return (b & (0x80 >> (x & 7))) ? 1 : 0;
}

// 在画布指定坐标绘制像素
static inline void set_canvas_pixel(uint8_t *canvas, int x, int y, uint8_t color) {
    if (x < 0 || x >= EPD_W || y < 0 || y >= EPD_H) return;
    int idx = y * (EPD_W / 8) + (x / 8);
    uint8_t mask = 0x80 >> (x & 7);
    if (color) {
        canvas[idx] |= mask;
    } else {
        canvas[idx] &= ~mask;
    }
}

// --------------------------- 三图层合成流水线 ---------------------------
void render_eye_to_canvas(int shiftX, int shiftY) {
    // 约束瞳孔移动幅度
    shiftX = constrain(shiftX, -MAX_OFFSET_X, MAX_OFFSET_X);
    shiftY = constrain(shiftY, -MAX_OFFSET_Y, MAX_OFFSET_Y);

    // ================= 图层 1：底层眼白 (固定底图) =================
#if HAS_SCLERA_LAYER
    memcpy(epd_canvas, gImage_sclera, BUFFER_SIZE);
#else
    memset(epd_canvas, 0xFF, BUFFER_SIZE);
#endif

    // ================= 图层 2：中层瞳孔 (依据 shiftX, shiftY 移动) =================
    int r2 = PUPIL_RADIUS * PUPIL_RADIUS;
    for (int dy = -PUPIL_RADIUS; dy <= PUPIL_RADIUS; dy++) {
        int dy2 = dy * dy;
        for (int dx = -PUPIL_RADIUS; dx <= PUPIL_RADIUS; dx++) {
            // 几何圆判断：仅提取圆内的瞳孔像素，圆形四周的黑色背景直接透明忽略
            if (dx * dx + dy2 <= r2) {
                int srcX = PUPIL_SRC_CENTER_X + dx;
                int srcY = PUPIL_SRC_CENTER_Y + dy;

                int dstX = srcX + shiftX;
                int dstY = srcY + shiftY;

                if (dstX >= 0 && dstX < EPD_W && dstY >= 0 && dstY < EPD_H) {
                    uint8_t pupilColor = get_bitmap_pixel(gImage_pupil, srcX, srcY);
                    set_canvas_pixel(epd_canvas, dstX, dstY, pupilColor);
                }
            }
        }
    }

    // ================= 图层 3：顶层眼眶与眼皮掩码遮罩 (固定顶层) =================
#if HAS_EYELID_LAYER
    // 采用字节级并行位运算：
    // mask 为 1 的位置是眼睑皮肤，替换为 gImage_eyelid；
    // mask 为 0 的位置是眼孔，保持底层画布（眼白+瞳孔）不变。
    for (int i = 0; i < BUFFER_SIZE; i++) {
        uint8_t m = gImage_eyelid_mask[i];
        epd_canvas[i] = (epd_canvas[i] & ~m) | (gImage_eyelid[i] & m);
    }
#endif
}

// --------------------------- 外部输入数据解析 ---------------------------
// 支持 JSON 格式 {"x": 10, "y": 5}、{x:10,y:5} 以及 CSV 格式 "10,5"
bool parse_input_coordinates(const String &str, int &outX, int &outY) {
    String s = str;
    s.trim();

    int xIdx = s.indexOf("\"x\"");
    if (xIdx < 0) xIdx = s.indexOf("x");
    int yIdx = s.indexOf("\"y\"");
    if (yIdx < 0) yIdx = s.indexOf("y");

    if (xIdx >= 0 && yIdx >= 0) {
        int colonX = s.indexOf(':', xIdx);
        int colonY = s.indexOf(':', yIdx);
        if (colonX > 0 && colonY > 0) {
            outX = s.substring(colonX + 1).toInt();
            outY = s.substring(colonY + 1).toInt();
            return true;
        }
    }

    int comma = s.indexOf(',');
    if (comma > 0) {
        outX = s.substring(0, comma).toInt();
        int secondComma = s.indexOf(',', comma + 1);
        if (secondComma > 0) {
            outY = s.substring(comma + 1, secondComma).toInt();
        } else {
            outY = s.substring(comma + 1).toInt();
        }
        return true;
    }

    return false;
}

// --------------------------- 蓝牙低功耗 (BLE) ---------------------------
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
BLECharacteristic *pCharacteristic = nullptr;

class MyCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pChar) override {
        std::string value = pChar->getValue();
        if (value.length() > 0) {
            String receivedData = String(value.c_str());
            int targetX = 0, targetY = 0;
            if (parse_input_coordinates(receivedData, targetX, targetY)) {
                posX = targetX;
                posY = targetY;
                newPositionReceived = true;
            }
        }
    }
};

void ble_init() {
    BLEDevice::init("ESP32_EPD_EYE");
    BLEServer *pServer = BLEDevice::createServer();
    BLEService *pService = pServer->createService(SERVICE_UUID);

    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );

    pCharacteristic->setCallbacks(new MyCallbacks());
    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->start();
    Serial.println("BLE advertising started.");
}

// --------------------------- 初始化与主循环 ---------------------------
void setup() {
    Serial.begin(115200);

    pinMode(EPD_BUSY, INPUT);
    pinMode(EPD_RST, OUTPUT);
    pinMode(EPD_DC, OUTPUT);
    pinMode(EPD_CS, OUTPUT);
    digitalWrite(EPD_CS, HIGH);

    SPI.begin(EPD_SCK, -1, EPD_MOSI, EPD_CS);
    SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE0));

    Serial.println("EPD init...");
    EPD_HW_Init();
    EPD_WhiteScreen_White();
    delay(500);

    // 初始居中合成眼球并写入局刷背景 RAM
    render_eye_to_canvas(0, 0);
    EPD_SetRAMValue_BaseMap(epd_canvas);

    ble_init();
    Serial.println("System Ready. All 3 Layers Active: Sclera + Moving Pupil + Eyelid Mask.");
}

void loop() {
    uint32_t now = millis();

    // 收到新坐标且超过保护间隔时触发局部刷新
    if (newPositionReceived && (now - lastRefreshTime >= MIN_REFRESH_INTERVAL_MS)) {
        newPositionReceived = false;

        if (posX != lastX || posY != lastY) {
            lastX = posX;
            lastY = posY;
            lastRefreshTime = now;

            // 1. 三层图像重新合成
            render_eye_to_canvas(posX, posY);

            // 2. 局部刷新推送
            EPD_Dis_PartAll(epd_canvas);
            partialRefreshCounter++;

            // 3. 累计 30 次局部刷新后全局重刷一次消除电泳残影
            if (partialRefreshCounter >= 30) {
                partialRefreshCounter = 0;
                EPD_HW_Init();
                EPD_WhiteScreen_ALL(epd_canvas);
                EPD_SetRAMValue_BaseMap(epd_canvas);
            }
        }
    }

    delay(5);
}
```

## File: tongkong.h
```c
#ifndef _TONGKONG_H_
#define _TONGKONG_H_

// 200x200, 1 bit/pixel, row-major, MSB first; 1=white, 0=black.
const unsigned char gImage_pupil[5000] = {
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X7F,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XD5,0X55,0X57,0XE0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0XFA,0X94,0X88,0X95,0X7E,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XA4,0XA5,0X25,0X22,0X4B,
  0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X3D,0X29,0X12,0X48,0X49,0X24,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XEA,0X92,0X94,0X92,
  0X92,0X4A,0X2E,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X03,0XD4,0X48,0X42,0X25,0X24,0X21,0X55,0X80,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0E,0XA2,0X85,
  0X28,0X40,0X11,0X0A,0X45,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X3D,0X54,0X30,0X12,0X0A,0X89,0X44,0X92,0XB8,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XEA,
  0X82,0X8A,0X91,0X48,0X50,0X21,0X29,0X5E,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XB4,0X54,0X2A,0X84,0X01,0X94,0X8A,0X44,
  0XA7,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X07,0X49,0X4A,0X88,0X41,0X54,0X42,0X21,0X12,0X29,0XC0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0E,0XA8,0X81,0X15,0X2A,0X04,0X28,
  0X88,0X45,0X26,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1A,0X95,0X2C,0X84,0XC1,0X21,0X80,0X82,0X10,0X92,0XB8,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X79,0X40,0X11,0X55,0X08,
  0X14,0X52,0X88,0X8A,0X49,0X5C,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XEA,0X2A,0X84,0X84,0X4A,0XA6,0X44,0XA0,0X21,0X04,0XAE,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XA5,0X04,0XA8,
  0X95,0X90,0X48,0XC0,0X45,0X04,0X52,0X17,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X03,0XA8,0XA1,0X0A,0X95,0X4B,0X14,0XAA,0XA8,0X52,0X52,
  0XAB,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XA4,
  0X54,0X54,0X41,0X10,0X52,0XA1,0X42,0X89,0X48,0X24,0XC0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0E,0X92,0X82,0X19,0X2A,0XBA,0X92,0XA5,0X24,
  0X55,0X04,0X92,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X1D,0X44,0X21,0X15,0X48,0X82,0X14,0XC1,0XA4,0X09,0X90,0X49,0X70,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X34,0X52,0X90,0X54,0X6D,0X58,0XA8,
  0XD5,0X2A,0XB3,0X49,0X4A,0X38,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X75,0X10,0X65,0X4A,0X0C,0X8A,0X04,0XC1,0X49,0X51,0X22,0X49,0X4C,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XE9,0X4A,0X91,0X4A,0X6C,
  0XA1,0X32,0XD9,0X44,0XA2,0XB1,0X52,0X5E,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X01,0XA8,0X2B,0X2A,0X2A,0XAC,0XA8,0X82,0XEB,0X2A,0X96,0X94,0X14,
  0X8B,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XCA,0X88,0X15,
  0X4A,0XAA,0XAA,0X2A,0XF7,0XCC,0XA3,0X4A,0X8B,0X2B,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X03,0X22,0X6A,0XB9,0X4E,0XCC,0XC4,0X93,0XEE,0XE9,0X8A,
  0XA0,0X92,0X55,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XAA,
  0X05,0X19,0X66,0X6E,0X52,0X43,0XFF,0XFA,0X46,0X15,0X22,0X25,0XC0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0D,0X4B,0X55,0X6B,0XA6,0X9E,0X69,0X1B,0XFF,
  0XFE,0XA6,0XD4,0X92,0X92,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0D,0X23,0X0D,0X6D,0XE9,0XDB,0X62,0X83,0XFF,0XFF,0X96,0X09,0X55,0X54,0X60,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1A,0XA8,0XD6,0XAA,0XE5,0XBA,0X95,
  0X2F,0XFF,0XFF,0XA9,0X49,0X0A,0XAA,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X3A,0X95,0XA2,0X7B,0X6C,0XDF,0X45,0X47,0XFF,0XFF,0XC8,0XA5,0X51,0XD1,
  0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X34,0XC4,0XD4,0XFB,0XEF,
  0X7E,0XD3,0X37,0XFF,0XFF,0XF2,0X17,0X5A,0XC5,0X38,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X69,0X2A,0XE2,0XDB,0X6F,0X9F,0XA9,0X17,0XFF,0XFF,0XF9,0X51,
  0X8D,0X54,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X6A,0X94,0X19,
  0XFF,0XEF,0XE7,0X9A,0X47,0XFF,0XFF,0XFC,0X94,0XDA,0XAD,0X2C,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0XD2,0X4A,0X97,0XEF,0X6F,0XF7,0X1A,0X6F,0XFF,0XFF,
  0XFE,0X55,0X6D,0X5B,0X6E,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XD1,
  0X3A,0XE2,0XEF,0XFF,0XFA,0X8B,0X4D,0XFF,0XFF,0XFF,0X27,0X3A,0XA5,0X96,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XAA,0XA1,0X2F,0XEF,0XFF,0XFE,0XE5,0X20,
  0X0F,0XFF,0XFF,0X28,0XAA,0XD2,0X67,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X01,0X95,0XAD,0X45,0XFF,0XFF,0XFE,0X29,0X54,0X80,0XFF,0XFF,0X95,0X59,0X2D,0X95,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0X50,0X45,0XCF,0XFF,0XFF,0XFC,
  0X55,0X24,0X94,0X1F,0XFF,0XC9,0X28,0XA9,0XA3,0X80,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X03,0X4E,0XAD,0X5F,0XFF,0XFF,0XF9,0X4A,0XA8,0XA5,0X07,0XFF,0XD2,0XD5,
  0X07,0X6D,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0X51,0X65,0X9F,0XFF,
  0XFF,0XF9,0X52,0X95,0X52,0X51,0XFF,0XE5,0X1A,0XBA,0X95,0X80,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X06,0X8A,0X6C,0XDF,0XFF,0XFF,0XF0,0X52,0X4A,0X94,0X48,0XFF,
  0XEA,0XED,0X5D,0XE5,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XA6,0XFD,
  0X3F,0XFF,0XFF,0XF6,0X99,0X22,0X61,0X55,0X3F,0XEB,0X2A,0X52,0X55,0XC0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X0D,0X33,0X17,0XBF,0XFF,0XFF,0XE3,0X44,0X00,0X15,
  0X08,0X1F,0XD4,0X82,0X9B,0XD4,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0E,
  0X55,0XBA,0XBF,0XFF,0XFF,0XE0,0XA0,0X00,0X02,0XD2,0X4D,0X2A,0X54,0X57,0XAA,0XA0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X09,0X56,0XD6,0X7F,0XFF,0XFF,0XCD,0X50,
  0X00,0X00,0X88,0X82,0XF5,0X26,0XBB,0X79,0X60,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X1C,0X2E,0XF3,0X7F,0XFF,0XFF,0XC4,0X00,0X00,0X00,0X45,0X53,0X54,0X8A,0XEF,
  0XEA,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1B,0X55,0X5D,0X7F,0XFF,0XFF,
  0XD3,0X00,0X00,0X00,0X29,0X4A,0XD0,0X3B,0X55,0X59,0X70,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X15,0X25,0X85,0XFF,0XFF,0XFF,0X88,0X00,0X00,0X00,0X02,0X2A,0XAD,
  0XE5,0X1D,0XAA,0X50,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X19,0X52,0XBA,0X54,
  0XBF,0XFF,0X85,0X00,0X00,0X00,0X15,0X54,0XF9,0XA6,0X66,0X59,0X70,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X1A,0X9B,0X44,0XF7,0X55,0XFF,0XA8,0X00,0X00,0X00,0X02,
  0XA0,0X2F,0XD2,0XA9,0XA4,0XB8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X34,0XA6,
  0X76,0XED,0XAE,0XAF,0X10,0X00,0X00,0X00,0X05,0X54,0X9A,0X09,0XA4,0X94,0X38,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X32,0XA6,0X88,0XBA,0XFB,0XD7,0X28,0X00,0X00,
  0X00,0X02,0X09,0X7A,0XA5,0X56,0XDB,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X34,0X99,0X6D,0XED,0X5E,0XB9,0X20,0X00,0X00,0X00,0X00,0XA5,0X42,0X95,0X0E,0XB4,
  0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X32,0X6C,0X51,0XFB,0XA5,0XAE,0X90,
  0X00,0X00,0X00,0X03,0XAA,0XAA,0XAB,0XA9,0X6D,0X58,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X2A,0XB6,0XD5,0XAD,0X7B,0X7A,0X40,0X00,0X00,0X00,0X00,0X15,0X55,0X2D,
  0X69,0X5E,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X75,0X6B,0XAC,0XF6,0X95,
  0XD6,0X00,0X00,0X00,0X00,0X00,0XA0,0X0F,0XE0,0XAA,0XF5,0X3C,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X2A,0X55,0X2B,0XED,0XFE,0XAA,0XA0,0X00,0X00,0X00,0X01,0X55,
  0XB5,0X5F,0X55,0XAA,0X98,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X69,0X2F,0X5B,
  0XFF,0XEB,0XBC,0X10,0X00,0X00,0X00,0X00,0X2D,0X41,0X25,0XD6,0XBA,0X6C,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X71,0X53,0X7C,0XFB,0X7D,0XFE,0XC0,0X00,0X00,0X00,
  0X01,0XA6,0XB4,0X48,0XA3,0XA9,0X2C,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X6D,
  0X49,0X43,0XED,0X47,0XEC,0X20,0X00,0X00,0X00,0X00,0X5A,0XAF,0XA5,0X6E,0X76,0XBC,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X62,0XFD,0XA9,0XB6,0XBB,0XB9,0X40,0X00,
  0X00,0X00,0X01,0X43,0X4D,0XFF,0XD2,0XEA,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X6A,0XAA,0X7D,0XFB,0XFF,0X5C,0X80,0X00,0X00,0X00,0X00,0XA8,0X2A,0X2A,0XFD,
  0XDD,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X68,0X57,0X45,0XDF,0X4F,0XFD,
  0X20,0X00,0X00,0X00,0X01,0X55,0X34,0X92,0XDE,0XE1,0X5C,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X6B,0X23,0X5B,0XA4,0XB7,0X54,0XA0,0X00,0X00,0X00,0X00,0X26,0XEA,
  0XD4,0XBD,0XFD,0X54,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X69,0XB5,0XEF,0X5B,
  0X7C,0XA8,0X40,0X00,0X00,0X00,0X01,0XAA,0XAA,0X5A,0X56,0XF5,0X5C,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X6A,0X46,0XBD,0XED,0XAA,0XED,0XA0,0X00,0X00,0X00,0X02,
  0X55,0X3E,0X85,0X5B,0XBC,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X71,0X33,
  0X76,0X97,0XD5,0XFA,0X48,0X00,0X00,0X00,0X01,0X44,0X4D,0X55,0XCA,0XD2,0XAC,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X6A,0XCD,0XDE,0XFC,0X5F,0X90,0XA0,0X00,0X00,
  0X00,0X04,0XB5,0X03,0XAA,0XA5,0XD6,0XDC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X29,0X5E,0XF3,0XAA,0XA3,0X45,0X28,0X00,0X00,0X00,0X06,0X5A,0XFE,0XE9,0X9E,0XAA,
  0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X75,0X2B,0XB8,0XF5,0XDD,0X56,0X90,
  0X00,0X00,0X00,0X01,0X4B,0X55,0XFE,0XD7,0XD5,0X5C,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X34,0XAE,0XAF,0X25,0X25,0XE8,0X54,0X00,0X00,0X00,0X15,0XA5,0X6A,0X2F,
  0XB5,0X52,0XB8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X2A,0XAA,0XC5,0XD0,0XB6,
  0X85,0XAA,0X00,0X00,0X00,0X0A,0X51,0XB5,0XD7,0XD7,0XAD,0X48,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X32,0X54,0XDE,0XEF,0XDF,0XBA,0XAD,0X00,0X00,0X00,0X2B,0X29,
  0X5F,0X5A,0XFB,0X75,0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X35,0X4B,0X6B,
  0XFF,0XFA,0XEA,0XA2,0X80,0X00,0X00,0X53,0X56,0XEA,0XC6,0XFE,0XEA,0X78,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X35,0X5A,0X95,0XBB,0XBA,0XAD,0X15,0X40,0X00,0X01,
  0X29,0XA6,0XEF,0X32,0X0F,0XD5,0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1A,
  0X4A,0XBD,0XEB,0XAA,0X95,0X6A,0X50,0X00,0X04,0XAC,0XB3,0X7D,0XAF,0XB3,0X2A,0XB0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1A,0XA9,0X4D,0XBE,0XEC,0X78,0X25,0X1A,
  0X00,0X12,0X65,0X4B,0X7E,0XD1,0X59,0XFC,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X1A,0X2B,0X73,0XFD,0XA3,0XA2,0XA9,0X55,0XA5,0X4B,0X2A,0X62,0XAB,0XAD,0X6D,
  0X65,0X70,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1A,0XA4,0XBF,0XBE,0XAD,0X55,
  0X65,0X54,0X29,0X49,0X91,0X99,0XEF,0XFB,0X25,0XDA,0XB0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X1D,0X16,0XE7,0XFB,0X55,0XA4,0XA4,0XA5,0XA5,0X55,0X4C,0X77,0X77,
  0XFB,0XD4,0X56,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0D,0X6A,0XB6,0XFD,
  0X6B,0X94,0XDD,0X16,0X55,0X54,0XEA,0X9F,0XB9,0X6E,0X6B,0XC1,0X60,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0D,0X29,0XBF,0XFA,0X9F,0X55,0XEA,0XAC,0XA5,0X4A,0X2B,
  0X25,0X6E,0XFF,0XB6,0XBC,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0D,0X4A,
  0X5D,0XB5,0X5D,0X23,0X42,0X31,0XB2,0X56,0X9D,0X9A,0XF7,0XAD,0XEA,0XA7,0X40,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XAA,0X96,0XF7,0XF6,0X5B,0X26,0XD4,0X95,
  0X2A,0X85,0X65,0X5F,0XAD,0X69,0X51,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X06,0X2B,0X5F,0XD6,0XFD,0XD6,0X8E,0X65,0XAA,0XA6,0XA6,0XD4,0XBF,0XDD,0XF6,0XAD,
  0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0X54,0XB6,0XBD,0X95,0X5D,0X75,
  0X8A,0XB1,0X6C,0X9A,0XA9,0XBF,0XEA,0XAB,0X45,0X80,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X03,0X4A,0X77,0XCA,0XBB,0XBA,0X56,0XC7,0X23,0X56,0X66,0XAC,0X4B,0X29,
  0X72,0XB5,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0X51,0XAE,0XB5,0XEB,
  0X69,0X97,0X55,0X4C,0XA5,0X96,0XAB,0X4E,0XFD,0X1A,0X97,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X01,0XAD,0X57,0XAA,0XFE,0XD5,0X6D,0X15,0X4A,0XD6,0XD3,0X6C,
  0XEF,0XAE,0XCD,0X55,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XA2,0XAB,
  0X5D,0X77,0XEB,0X4C,0X56,0XA6,0X56,0X49,0X9B,0X0D,0XFF,0XAA,0XAB,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XDA,0XA8,0XAA,0XFB,0X35,0X3E,0XAB,0X55,0X5A,
  0XB6,0XAD,0X6E,0X72,0XAD,0X56,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0XD5,0X55,0XF1,0XB7,0XF7,0X55,0X17,0X13,0X2E,0X65,0X6F,0XAD,0X59,0XA5,0X2E,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X64,0XAD,0X4E,0X7E,0XB4,0XFA,0XCA,
  0XA6,0X6B,0XB5,0X55,0X6E,0XB5,0XD2,0XDC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X72,0XA9,0X75,0XDF,0XCE,0XB8,0X67,0X13,0X39,0X96,0XDE,0XD5,0XBA,0X5A,
  0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X2D,0X15,0X5B,0XFF,0X69,
  0XEA,0X95,0X55,0X4F,0XC9,0X6B,0XAE,0X9D,0XD5,0X58,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X3A,0XCB,0X4D,0XB6,0XAD,0XB5,0XAE,0X2B,0X1A,0XFB,0XB6,0XF6,
  0XE3,0X0A,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1A,0X21,0X6B,
  0XBE,0XBD,0XF6,0XA7,0X53,0X4D,0X3D,0X7E,0XDD,0X5F,0X6A,0XF0,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X0D,0X5A,0X2D,0X95,0X6B,0XB4,0X9C,0X2A,0XAF,0XCA,
  0XCF,0X76,0X94,0XAD,0X60,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0D,
  0X45,0XBA,0X5F,0XD7,0X75,0X57,0X4A,0X96,0XE9,0XEF,0X59,0X69,0X52,0XC0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0X52,0XA3,0XF6,0XDB,0XEA,0X9A,0X6A,
  0X9B,0XD5,0X33,0XD6,0XAB,0X2B,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X03,0X2A,0XAB,0X66,0XDE,0XC5,0X7B,0X4D,0XEE,0XED,0XAB,0XFE,0XAD,0X55,0X80,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XA4,0X26,0X8A,0XBD,0XF7,
  0X69,0X55,0X6B,0XB5,0X96,0XD2,0XB2,0XAF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X01,0XD5,0X5B,0X3D,0X77,0X99,0X6E,0X95,0X56,0XB4,0XBB,0XCD,0X54,
  0X56,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XE5,0X56,0XA4,
  0XAD,0XD6,0XDA,0XDA,0XDB,0XAE,0XCB,0X75,0XB6,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X74,0X92,0XAB,0XF5,0XAA,0XCD,0XAD,0X6D,0XF7,0X56,
  0XAE,0XAB,0X5C,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3A,
  0XAD,0X57,0X4D,0XB5,0XBD,0X6E,0XD6,0XB5,0XA9,0X77,0XAA,0XB8,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1D,0X52,0X39,0XD7,0X4A,0XF5,0X4C,0X5A,
  0XEC,0XAB,0X52,0XAA,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X0E,0XA8,0XA5,0X29,0XAB,0XB5,0XB7,0X6B,0XB7,0X35,0X9A,0X55,0X60,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0X55,0X5C,0XB7,0X5C,0XD6,
  0X46,0XD5,0X6D,0X56,0XA6,0XAB,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X03,0XA4,0X82,0X66,0XD5,0XEA,0XDA,0XAF,0X7E,0XAB,0X55,0X55,0X80,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XD5,0X6A,0XBC,
  0X99,0XAD,0X4D,0X15,0X67,0X59,0X54,0XB6,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X7A,0X92,0XAF,0XCB,0XAA,0XA3,0XF9,0X56,0XED,0X53,
  0X5C,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X35,
  0X55,0X42,0XD5,0X55,0X6A,0X4D,0XEB,0X22,0X4A,0XB8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1D,0X52,0XAA,0XFD,0X53,0X25,0X55,0X3D,
  0XAA,0XAB,0X70,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X07,0X4A,0X95,0X56,0XA5,0XB6,0XD7,0X55,0X52,0XAA,0XC0,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XAA,0X54,0XAA,0XAD,0X22,
  0X95,0X25,0XA9,0X57,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XEA,0XA5,0X52,0XA5,0X5B,0X55,0X5A,0X6A,0XAE,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7A,0X54,0XAA,
  0X95,0X22,0X4C,0XA5,0XAA,0XBC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X1E,0XAA,0X4A,0X2A,0X95,0X52,0X95,0X2A,0XF0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,
  0XA5,0X54,0XAA,0X92,0X14,0XCA,0XAB,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XF5,0X52,0XA4,0XA9,0XA5,0X25,0X5F,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X7A,0XAA,0X54,0X94,0X54,0XAA,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XAA,0X92,0XA3,0X55,
  0X57,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X01,0XFA,0XAD,0X5A,0XAA,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XEA,
  0XAA,0XB7,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0X7F,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X01,0X54,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00
};

#endif // _TONGKONG_H_
```

## File: yanbai.h
```c
#ifndef _YANBAI_H_
#define _YANBAI_H_

// 200x200, 1 bit/pixel, row-major, MSB first; 1=white, 0=black.
const unsigned char gImage_sclera[5000] = {
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0X40,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,
  0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0X77,
  0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XEF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XEF,
  0XBF,0XFF,0XFF,0XFF,0XFD,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XBD,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0XFE,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XF7,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X0F,0XBF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3E,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0XFF,0XF7,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XBF,0X80,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XF7,0XFF,0XBF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFB,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X06,0XDE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X78,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X3B,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XEE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X6F,
  0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XF7,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X03,0XBF,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0XE0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XEE,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3A,0XFB,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X5F,0XBF,0XBF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XEB,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X01,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFD,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0X9B,
  0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0X60,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X05,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0D,0XED,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,
  0XB8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X16,0XBF,0XFD,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X3B,0XF7,0XF7,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFE,0XF4,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X55,0X7F,0X7F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFB,0XDE,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XD7,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XEB,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0X6D,0X7F,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XBD,0X80,0X00,
  0X00,0X00,0X00,0X00,0X00,0X03,0XB7,0XF6,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF6,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X04,0XAA,
  0XBF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0XBB,
  0X60,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0X6E,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0XFD,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,
  0X13,0XEB,0XDB,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XDF,0XEE,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,0X1D,0XB6,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X76,0XDC,0X00,0X00,0X00,0X00,
  0X00,0X00,0X17,0XDB,0XFE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFD,0XFB,0X56,0X00,0X00,0X00,0X00,0X00,0X00,0X1A,0XF6,0XAF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0XBD,0XBF,0X00,0X00,
  0X00,0X00,0X00,0X00,0X0F,0X5B,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XBF,0XEB,0X67,0X80,0X00,0X00,0X00,0X00,0X00,0X03,0XF5,0X6F,
  0XF7,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X7E,0XFD,
  0XC0,0X00,0X00,0X00,0X00,0X00,0X01,0XBD,0XFF,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFB,0XFF,0XD7,0X5E,0XE0,0X00,0X00,0X00,0X00,0X00,0X01,
  0XD6,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,
  0X7A,0XFF,0X60,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0X7D,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0XEF,0XEF,0XA0,0X00,0X00,0X00,0X00,
  0X00,0X00,0X3B,0XF7,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XDF,0XBB,0XBE,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0X5F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0XFB,0XFB,0XC0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0X56,0XDF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XEF,0XBF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XED,0XFF,0XFC,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XAF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0XFE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XBB,0X7F,
  0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFB,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1F,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XB5,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XEF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFE,0XDC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFB,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XEF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0F,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1E,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,
  0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X17,0XFF,0XFD,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00
};

#endif // _YANBAI_H_
```

## File: yankuang.h
```c
#ifndef _YANKUANG_H_
#define _YANKUANG_H_

// 200x200, 1 bit/pixel, row-major, MSB first; 1=white, 0=black.
const unsigned char gImage_eyelid[5000] = {
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X15,0X6E,0XA0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0B,0XFF,
  0XFB,0XFF,0XA0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XD7,0XBF,0X7B,0XFF,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,
  0X7E,0XFE,0XED,0XD6,0XDF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFD,0XDB,0XDB,0X56,0XBF,0XFA,0XDF,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0F,0X7F,0XB7,0X75,0XEB,0XDD,0X2D,0XBF,0XD0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3D,0XD5,0XFD,0XDF,0XBF,0XEA,0XD5,
  0X69,0X78,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X01,0XFF,0XF7,0X57,0X7B,0XFF,0X57,0XFF,0XBF,0XAF,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XF7,0X6D,0XFF,0XD6,0XDA,
  0XFE,0XAA,0XD2,0XFB,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X1F,0XBD,0XDF,0XBF,0XFF,0XF7,0XFB,0XED,0X6D,0X16,0XF8,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7E,0XF7,0XBE,0XFB,
  0X7E,0XAD,0XAE,0XBF,0XDD,0XFA,0XAE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X01,0XF7,0XFF,0X7B,0XFF,0XF7,0XDE,0XF7,0X69,0XEB,0X56,0XB5,
  0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XDD,
  0X7F,0XED,0XBB,0X7B,0XAD,0XFF,0X2D,0X6B,0X5B,0X40,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XDE,0XFF,0XFF,0XFF,0XEE,0XF6,0XDF,0X5A,0XFB,
  0XBD,0X6D,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,
  0XFB,0XF7,0XFF,0XFB,0XB7,0XDB,0X75,0XAB,0XAD,0X55,0X55,0X3C,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFB,0X6F,0XFF,0XF7,0XBC,0XDA,0XAF,0XFE,
  0XEF,0XF5,0XD6,0XDA,0XD6,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X01,0XFE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFA,0XAB,0X54,0X57,0X6B,0X55,0X5B,0X80,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFB,0XFF,0XF6,0XFF,0XFF,0XFF,
  0XFF,0XF5,0XFB,0X95,0XBA,0XB5,0X6D,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X0F,0XFF,0XFF,0XDF,0XEF,0X7B,0X6A,0X95,0X5E,0XAD,0X6A,0XCB,0X55,0X55,
  0X70,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XAF,0XFF,0X7F,0XFF,
  0XEF,0XFF,0XEA,0X82,0XA2,0X92,0X2D,0X6D,0XAA,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X7F,0XFE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0X5D,0X6D,0X52,
  0XAB,0X6A,0XBC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFE,0XFF,0XF7,
  0XFF,0XFF,0XFE,0XFF,0X5F,0XFF,0XF7,0XFF,0X6C,0X94,0X96,0XA7,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X01,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XDD,0XFD,0XF9,0X54,
  0X02,0XF5,0X6A,0XA9,0X5A,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X57,0XFB,0X59,0X15,0XB6,0XD5,0X55,0XC0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0X5F,0XED,0X44,0X55,0X2A,0XD6,0XA0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X1F,0XBF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XBD,0X76,0XA8,0XB5,0X29,
  0X70,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XF7,0XFF,0XFF,0XFF,0XFF,0XEF,
  0XFF,0XFF,0XFF,0XF5,0X6B,0XFB,0XD6,0X95,0X56,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X7F,0XFF,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X7D,0X54,0X2A,0XA9,
  0X6A,0XAA,0XD4,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFD,0XEF,0XEF,0XFF,0XEF,0X8A,0XDD,0XAA,0XA5,0X5F,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X01,0XFD,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XBD,0XBD,0XEF,0XFB,
  0XEB,0X26,0X52,0XBA,0X52,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XF6,0XD7,0X55,0X5E,0XF6,0XD9,0XAD,0X4B,0X4B,0X80,0X00,
  0X00,0X00,0X00,0X00,0X00,0X07,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XDD,0XAD,0X5A,0XA8,
  0X52,0XA9,0X2D,0XAC,0XAA,0XA9,0X75,0X60,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XEF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X7E,0XA0,0X57,0XFF,0XFF,0X52,0X8A,0XA2,0XAA,0XAA,0X95,
  0X60,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFA,0XA0,0X97,
  0XAD,0X5F,0XFF,0XFE,0XB0,0XBD,0X55,0X55,0X6A,0XB0,0X00,0X00,0X00,0X00,0X00,0X00,
  0X3F,0XBF,0XFF,0X7F,0X7F,0XFF,0XE8,0X17,0XFF,0XFF,0X60,0X57,0XFF,0XEE,0X45,0X25,
  0X55,0XAB,0X58,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFE,0XFF,0XFF,0XFF,0XFE,0XA3,
  0XFF,0XFF,0XFF,0XFF,0X88,0XB7,0XFB,0XD5,0XDA,0XAA,0XAC,0XAC,0X00,0X00,0X00,0X00,
  0X00,0X00,0XFE,0XFF,0XEF,0XFF,0XF7,0XAA,0X0F,0XFF,0XFF,0XFF,0XFF,0XF4,0X4A,0XAE,
  0X5C,0X55,0X54,0XAB,0X6A,0X00,0X00,0X00,0X00,0X00,0X00,0XFB,0XFF,0XFF,0X7F,0XFD,
  0X61,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0X81,0X7B,0XAB,0X52,0XAB,0X55,0X5F,0X00,0X00,
  0X00,0X00,0X00,0X01,0XFF,0XF7,0XFF,0XED,0X6E,0X8F,0XFF,0XFF,0XFF,0XFF,0XDF,0XFF,
  0XFE,0X0F,0XEF,0XDC,0X25,0X55,0X52,0X80,0X00,0X00,0X00,0X00,0X03,0XEF,0XFE,0XFB,
  0XFF,0XB0,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0XB0,0X67,0X95,0X6A,0XEB,
  0X80,0X00,0X00,0X00,0X00,0X07,0XFF,0X6F,0XEF,0XDE,0XCA,0XBE,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFB,0XF4,0X5F,0X38,0XEA,0XAA,0XAD,0X40,0X00,0X00,0X00,0X00,0X07,0XFD,
  0XFF,0XBE,0XFB,0X25,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0XFE,0XFA,0X0B,0XA6,0X75,
  0X56,0XAA,0XE0,0X00,0X00,0X00,0X00,0X0F,0XBF,0XFF,0XFB,0XA8,0X1F,0XEF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XA1,0XD7,0X0D,0X52,0XD7,0X40,0X00,0X00,0X00,0X00,
  0X1F,0XF7,0XDA,0XDD,0X51,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XBF,0XDB,0XD4,
  0X73,0XD5,0XAB,0X79,0X70,0X00,0X00,0X00,0X00,0X3E,0XFF,0X7F,0XA5,0X42,0XB7,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0XFE,0XF4,0X1C,0XEA,0XBD,0X16,0XB8,0X00,0X00,
  0X00,0X00,0X3F,0XDB,0XFE,0XFA,0X1D,0XFF,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0X7F,0XFD,
  0X77,0XBB,0X05,0X3A,0XA6,0XEB,0XA8,0X00,0X00,0X00,0X00,0X7B,0XFE,0XEB,0XA8,0X3B,
  0XFF,0XFF,0XFF,0XFF,0XDF,0XF7,0XFF,0XEF,0XFF,0XDE,0XED,0XA2,0XAB,0X55,0X55,0X6C,
  0X00,0X00,0X00,0X00,0X7E,0XBF,0XAE,0XD1,0X6F,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XBE,0XA8,0XD5,0X7A,0XAD,0XB4,0X00,0X00,0X00,0X00,0XF7,0XFB,0XF9,
  0X03,0XBF,0XFF,0XFF,0XFF,0XDF,0XFF,0XFF,0XFB,0XFF,0X7F,0XFE,0XF7,0X68,0X57,0X4F,
  0X6A,0XAE,0X00,0X00,0X00,0X01,0XFF,0XED,0X56,0X8D,0XFF,0XDF,0XFB,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFD,0XFB,0XF7,0XFD,0XB6,0X29,0X52,0XAB,0X6F,0X00,0X00,0X00,0X01,0XD6,
  0XDB,0X54,0X16,0XEF,0XFF,0XFF,0XFB,0XFF,0XFF,0X73,0XFE,0XFF,0XFF,0XFB,0XD7,0X69,
  0X14,0XAA,0XDA,0XB3,0X00,0X00,0X00,0X03,0XFF,0XFF,0XA8,0X2F,0XFF,0X7F,0XFD,0XFF,
  0XEF,0XF5,0XFB,0XBF,0XFF,0XFF,0X7D,0XFD,0XDE,0X87,0X35,0X57,0X5D,0X00,0X00,0X00,
  0X03,0XB5,0X54,0XA0,0XDB,0XFF,0XF7,0XFE,0XFD,0XD7,0XF7,0XFA,0XEC,0XF3,0XFF,0XBF,
  0X7F,0XB7,0X61,0X8E,0XAD,0X65,0X80,0X00,0X00,0X07,0XEF,0XEA,0X01,0X7F,0X5F,0XBF,
  0XFE,0X7E,0X7A,0XED,0X99,0XD9,0XEF,0X7F,0XBB,0XFA,0XFA,0XA8,0XF1,0X53,0XBB,0XC0,
  0X00,0X00,0X07,0X5F,0XB5,0X45,0XBF,0XFD,0XDF,0XF7,0X7B,0X70,0X96,0XE5,0XE5,0X56,
  0XFE,0X7F,0XBF,0XAF,0XD0,0X5D,0X5C,0XAC,0XC0,0X00,0X00,0X0F,0XFA,0XA8,0X0A,0XED,
  0X7E,0XFF,0XDB,0X8B,0X1A,0X2A,0X4C,0X71,0X6C,0XBE,0XFB,0XED,0XF5,0X6C,0X45,0X6B,
  0X6B,0X60,0X00,0X00,0X0E,0XAD,0X52,0X97,0XFB,0XEB,0XEF,0X7F,0XA0,0X84,0X96,0X92,
  0X1A,0XE4,0XB8,0XF7,0XFF,0X5D,0XB6,0X39,0X55,0XAA,0XA0,0X00,0X00,0X1D,0XF6,0XA2,
  0X1B,0XBF,0XFF,0XF9,0XFE,0XCD,0X28,0X2C,0X4D,0X04,0X25,0X51,0XB7,0XF6,0XFE,0XD5,
  0X8E,0XAA,0XB5,0X70,0X00,0X00,0X1F,0XBA,0X88,0X6E,0XEE,0XBF,0XF5,0XE9,0X25,0X29,
  0X49,0X52,0X41,0X52,0X43,0X67,0XAD,0XFB,0X75,0X45,0X57,0X5D,0X50,0X00,0X00,0X3A,
  0XCA,0XA9,0X77,0XBB,0XEF,0XBE,0X40,0X0A,0X22,0X48,0X8B,0X34,0X20,0X0D,0X9B,0X7B,
  0XDF,0XAE,0X43,0X94,0XEB,0X68,0X00,0X00,0X3F,0X75,0X21,0X5D,0XDA,0XFE,0XF5,0X04,
  0X54,0X44,0X91,0X48,0X88,0X8A,0X15,0X0B,0XE7,0XF6,0XF5,0X60,0XEB,0X35,0XB8,0X00,
  0X00,0X35,0XB4,0X85,0XF7,0X5F,0XBD,0XF4,0X80,0X40,0X08,0X25,0X4A,0X25,0X49,0X50,
  0X92,0X9F,0XEF,0XBD,0XB0,0XB5,0X96,0XA8,0X00,0X00,0X7F,0XD2,0X45,0X5D,0X6B,0XD3,
  0XAA,0X22,0X91,0X12,0X4B,0X5B,0XB4,0X08,0X0C,0X21,0X3F,0X7A,0XEB,0XA8,0X2A,0XD5,
  0XAC,0X00,0X00,0X6A,0XAA,0X97,0XF6,0XF6,0XBE,0XA0,0X09,0X10,0X51,0X5C,0X20,0X4B,
  0XE6,0XA0,0X52,0X55,0XF5,0XDE,0XF4,0X55,0X35,0XBC,0X00,0X00,0XDD,0X54,0X15,0X5B,
  0XFA,0X6D,0X01,0X18,0X45,0X45,0X00,0X00,0X00,0X35,0X02,0X89,0X5B,0XEF,0XF3,0X56,
  0X16,0XAE,0XCA,0X00,0X00,0XF6,0XAA,0X2E,0XEE,0XD9,0X10,0X00,0X61,0X00,0X58,0X00,
  0X00,0X00,0X03,0XE9,0X48,0X24,0X37,0X5F,0XEA,0X8B,0X52,0XB6,0X00,0X00,0XDA,0XA8,
  0X5B,0X95,0XAE,0X40,0X04,0XC2,0X6B,0X00,0X00,0X00,0X00,0X00,0X34,0X25,0X11,0X5F,
  0XF6,0XB5,0X89,0XED,0X5A,0X00,0X00,0XED,0X52,0X6E,0XB2,0XBD,0X80,0X95,0X04,0X94,
  0X00,0X00,0X00,0X00,0X00,0X07,0X92,0X2A,0X17,0X7F,0X7C,0XC4,0X75,0XAF,0X00,0X01,
  0XB6,0XA8,0XDB,0X7C,0XC2,0X44,0X10,0X12,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0XE3,
  0X01,0X2A,0XDB,0XAB,0X47,0X16,0XD3,0X00,0X01,0XEA,0XA1,0XBC,0XD7,0X74,0X89,0X01,
  0X6A,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1C,0X91,0X55,0XFB,0XDE,0XB1,0XAA,0XBD,
  0X00,0X01,0XB5,0X82,0XEB,0X7F,0X28,0X30,0X26,0X90,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X06,0X88,0X2A,0XBE,0XF7,0X68,0XAE,0XAB,0X80,0X03,0XDA,0X57,0X71,0X89,0X80,
  0X04,0X5C,0XA0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XD5,0X05,0X6F,0XB9,0XA8,
  0X63,0X55,0X00,0X03,0X6D,0X45,0XD6,0XFC,0X01,0X00,0XA2,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X64,0X26,0XB6,0XEE,0XD4,0XB9,0XEB,0X80,0X03,0XAA,0X8F,0X57,
  0X70,0XA4,0X12,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X19,0X11,0X5B,
  0XFF,0X6A,0X16,0X3A,0X80,0X03,0X6A,0X1A,0XCA,0X82,0X00,0X85,0X40,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X06,0X52,0X6D,0XDB,0XB6,0X4D,0X97,0XC0,0X07,0X55,
  0X2D,0X7D,0X01,0X01,0X2A,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,
  0X11,0X56,0XFD,0X6B,0X26,0XED,0X40,0X05,0XB4,0X77,0X10,0X38,0X12,0X14,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X88,0X2B,0X6F,0XFD,0X52,0XB2,0XC0,
  0X06,0XD2,0X5D,0X83,0XA0,0X04,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X65,0X15,0X5A,0XCB,0X4A,0XAD,0X40,0X07,0X49,0X60,0X2C,0X01,0X48,0X50,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X2A,0X00,0XAF,0XDE,0XA5,
  0X55,0X60,0X05,0X54,0X95,0XA0,0X02,0X13,0X40,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X0A,0XCB,0X77,0XF6,0XEA,0XB6,0XC0,0X0E,0XAB,0XFA,0X48,0X24,
  0X22,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X05,0X20,0X95,
  0XAB,0X55,0X5A,0XE0,0X06,0XA5,0XA7,0XA1,0X48,0X4D,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X02,0X68,0X56,0XFD,0XF5,0X4A,0XA0,0X0D,0X54,0X28,
  0X02,0X10,0X0A,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,
  0X93,0X5B,0X7E,0XAD,0X75,0X60,0X0D,0X57,0X80,0X04,0X41,0X34,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X98,0X4D,0XBB,0XFA,0X95,0X60,0X0D,
  0XAA,0XFD,0X92,0X08,0XA8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X65,0X26,0XDF,0X6B,0X57,0X50,0X0E,0XD6,0X92,0X28,0X12,0XA0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X14,0X95,0X57,0XB7,0XA9,
  0XE0,0X0A,0XAA,0XD4,0X20,0X41,0X50,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X1A,0X4A,0XEB,0XDE,0XD6,0X70,0X0F,0X6E,0X40,0XD2,0X4A,0XA0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X05,0X45,0X5E,
  0XF5,0X6A,0XA0,0X1D,0XBB,0X02,0X80,0X12,0X40,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XB2,0XB5,0XEF,0XF5,0X70,0X0E,0XEC,0X1E,0X09,
  0X4A,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X02,
  0X88,0XAB,0XFD,0X5A,0XB0,0X1A,0X80,0XF0,0XA0,0X2A,0X80,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0X54,0X5F,0X7F,0XEE,0XB0,0X0F,0X6A,
  0X82,0X0A,0X53,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0XA2,0X85,0X7A,0XDE,0XB0,0X18,0X90,0X05,0X10,0X54,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X54,0X55,0XFF,0XF5,0X50,
  0X1F,0XEF,0X6A,0X22,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X2A,0X96,0XAD,0XBF,0XD0,0X1B,0X52,0XB4,0X00,0XA8,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X15,0X69,0XDF,
  0XF5,0XF0,0X1D,0X55,0X40,0X95,0X54,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X1A,0X8A,0XBF,0XDE,0XB0,0X0D,0X5E,0XA9,0X21,0X7C,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0A,
  0X52,0XBF,0X7B,0XF0,0X1D,0XAA,0XA0,0X4A,0XAC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X05,0X2A,0X6B,0XFF,0X70,0X16,0XEA,0X44,
  0X92,0XEC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X06,0X8A,0XAF,0XDF,0X70,0X1D,0X55,0X12,0X95,0X7C,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X02,0XA5,0X57,0XFF,0XD0,0X0D,
  0XAA,0X81,0X2A,0XF6,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X03,0X50,0XAF,0XED,0XF0,0X1A,0XA8,0X2A,0X95,0X5E,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X06,0XAA,0XAB,0XFF,
  0X70,0X0E,0XAA,0X95,0X55,0X7D,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X07,0XC9,0X09,0XBD,0XF0,0X0A,0XB2,0X52,0XAB,0XEF,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0D,0X5E,
  0XE5,0XFF,0XF0,0X0E,0XC9,0X55,0X54,0XBF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XAB,0X31,0XFF,0XF0,0X0D,0X16,0XAA,0XB7,
  0X6F,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0E,0XB6,0XCC,0XBF,0XB0,0X0C,0XAA,0XAA,0X42,0XF7,0XE0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XBF,0X54,0XFF,0XF0,0X0A,0XEB,
  0X6A,0XA9,0XBD,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1D,0XEF,0XEA,0X3E,0XE0,0X0D,0X9D,0XAD,0XB6,0X77,0XF8,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFE,0XBD,0X3F,0XF0,
  0X06,0XEB,0X7A,0X92,0XDF,0X7C,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X7D,0XDB,0XFA,0XBF,0XE0,0X0F,0X7D,0XA0,0X2A,0X5D,0XFE,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XBF,0XF5,
  0X1F,0XE0,0X06,0XB7,0X4F,0XA1,0X6B,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0XF7,0X7F,0X9C,0X8F,0XE0,0X07,0XFD,0X55,0X5E,0XAF,
  0XEF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,
  0XFF,0XFE,0XAF,0XE0,0X07,0X6B,0XBA,0XB5,0X5F,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFD,0XFF,0XFA,0X6F,0XC0,0X07,0XFE,0XD6,
  0XC0,0XA1,0X7F,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X03,0XEF,0X7F,0XFE,0X8B,0XC0,0X03,0XF6,0XDD,0X2A,0XAF,0XFF,0XFC,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XD7,0XAF,0XC0,0X03,
  0XBD,0XB4,0XDF,0XB5,0X7D,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1F,0XBF,0XFB,0XFD,0X57,0XC0,0X03,0XFB,0XEB,0XAA,0X12,0XAF,0XFF,0XC0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XF7,0XFF,0XE8,0X2F,
  0X80,0X03,0XDF,0X5A,0XBC,0XCD,0X7F,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0X26,0XDF,0X80,0X01,0XF7,0XB6,0XE5,0X37,0X83,
  0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFE,0XAE,0XA1,
  0X7B,0XB7,0X80,0X01,0XFE,0XFD,0XB4,0XD4,0X7C,0XBF,0XFF,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XF5,0X53,0XBF,0XDE,0XFF,0X80,0X01,0XFE,0XD7,0X5B,
  0X02,0XA7,0X7F,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0X95,
  0XAD,0XDF,0X77,0XFF,0X00,0X01,0XFB,0XED,0XF5,0X3B,0X7D,0XFF,0XFF,0XF0,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFE,0XBF,0XFE,0XFB,0XFE,0XFF,0X00,0X00,0XFF,
  0XFF,0X58,0X5F,0X95,0X2B,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,
  0XFA,0XF6,0XFF,0XBF,0XDF,0XFF,0X00,0X00,0XF7,0X5B,0X63,0XEA,0XF1,0XD7,0XFF,0XFF,
  0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFD,0X4F,0XFF,0XF7,0XFF,0XFF,0XFE,0X00,
  0X00,0XFF,0XEE,0XCE,0XBF,0XCF,0XDB,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X1F,0XF7,0XE3,0XAF,0XDF,0XFF,0XFF,0XFE,0X00,0X00,0X7E,0XFF,0XB6,0XEB,0X1C,0X4F,
  0XBF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X03,0XFF,0XDE,0XFD,0XF7,0XFD,0XFF,0XFF,
  0XFC,0X00,0X00,0X7F,0XB6,0X5B,0X77,0X63,0X38,0X8B,0XFF,0XFF,0XF8,0X00,0X00,0X00,
  0X00,0X7F,0XFE,0XB7,0XFF,0X7B,0XBF,0XFF,0XFF,0XFC,0X00,0X00,0X3F,0XFD,0XFE,0XDC,
  0X7C,0XF2,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X7F,0XFF,0XFF,0XFA,0XBF,0XFF,0XFF,
  0XFF,0XFF,0XFC,0X00,0X00,0X3F,0XFB,0XAB,0XF9,0XC5,0XC7,0X79,0X5F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XD9,0XBF,0X5F,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X3F,0XF7,
  0X7D,0X57,0X3F,0X1D,0XEE,0XAF,0XFF,0XDF,0XFF,0XFF,0XFF,0XFF,0XFD,0X3E,0XEF,0XFF,
  0XEF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X1F,0XBF,0XEF,0XDA,0XF4,0X7E,0XA1,0X7D,0XF7,
  0XFF,0XFF,0XFF,0XF7,0XB7,0X37,0XFF,0X77,0XFF,0XFE,0XFF,0XFF,0XFF,0XF0,0X00,0X00,
  0X1E,0XFE,0XFD,0X7B,0XDD,0XEB,0XCB,0X75,0X5F,0XFD,0XFF,0XEF,0XFF,0XF7,0XC7,0XDF,
  0X7D,0XF7,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X0F,0XF7,0XB7,0XD5,0X7B,0XFD,0X8D,
  0XE5,0XFE,0XBB,0XDE,0XFE,0XF7,0XD7,0XFB,0XF7,0XBF,0XFF,0XF7,0XFF,0XFF,0X7F,0XF0,
  0X00,0X00,0X0F,0XFE,0XFD,0X77,0XF7,0XB7,0X3B,0XD7,0X17,0X66,0XDC,0XB6,0XF7,0XFF,
  0XB9,0XDB,0XDF,0XF7,0XFF,0XDF,0XF6,0XFF,0XE0,0X00,0X00,0X0F,0XFF,0XEF,0XD5,0XAE,
  0XEE,0X6E,0X8E,0X78,0XCF,0XDD,0XAF,0XF7,0X7B,0XDE,0XFF,0XFF,0XFB,0XFF,0XFF,0XFF,
  0XFF,0XE0,0X00,0X00,0X07,0XDB,0XBF,0X6F,0XEF,0XFC,0XFB,0XB8,0XDD,0X98,0XDB,0X6E,
  0XE7,0XF9,0XEB,0X7D,0XEF,0XDF,0XFF,0XFF,0X6F,0XFF,0XC0,0X00,0X00,0X07,0XFF,0XF5,
  0XDF,0XBD,0XBB,0XDF,0XA7,0XF9,0X27,0XDA,0XDF,0XCE,0XB5,0XF7,0XFF,0XFF,0X7F,0XFB,
  0XFF,0XBD,0XDF,0XC0,0X00,0X00,0X03,0XFF,0XFF,0XFD,0XFF,0XED,0XFF,0X5F,0X73,0X6F,
  0X7A,0XB6,0X6F,0XDE,0XDB,0XF7,0XF6,0XFB,0XEF,0XED,0XFF,0XFD,0X80,0X00,0X00,0X01,
  0XFF,0XBB,0X77,0X77,0XEB,0XDF,0X3B,0XF2,0X5B,0XB6,0XAD,0XEB,0X7E,0X7D,0XDE,0XBF,
  0XFF,0XBF,0X7E,0XFF,0XFF,0X80,0X00,0X00,0X01,0XFF,0XFF,0XDF,0XDF,0X5F,0X7A,0X7F,
  0XE6,0XBF,0X6D,0XEF,0X4F,0XD7,0XFD,0XFF,0XF7,0XFF,0XDD,0XD5,0XEF,0XEF,0X00,0X00,
  0X00,0X00,0XFF,0XED,0XFD,0XBD,0XB7,0XFC,0XEF,0X6E,0XFF,0X6E,0XDF,0X6E,0XEF,0XFF,
  0X57,0XFF,0XDB,0X7F,0XAF,0XFE,0XFE,0X00,0X00,0X00,0X00,0XFE,0XFF,0XB7,0XFF,0XFF,
  0XDD,0XFD,0XE6,0XEF,0X7E,0XFB,0XDF,0X7E,0XB5,0XFF,0XFB,0X7E,0XF6,0XBF,0XDB,0XFE,
  0X00,0X00,0X00,0X00,0X7F,0XFF,0XFE,0XEB,0X76,0XF5,0XDF,0XED,0XFD,0XEE,0XDF,0XB5,
  0XFF,0XFE,0XFE,0XDF,0XFB,0XDD,0X7B,0X7F,0XDC,0X00,0X00,0X00,0X00,0X7F,0XFE,0XDF,
  0XFF,0XFF,0XFB,0XFD,0XED,0XFE,0XDE,0XFD,0XBF,0XDB,0XD3,0XFB,0XFB,0X5D,0X6B,0XFE,
  0XEF,0XF8,0X00,0X00,0X00,0X00,0X3F,0XDF,0XFB,0X5D,0XD7,0XFF,0XFF,0XDD,0XF7,0XFD,
  0XF7,0XAF,0XFF,0X6F,0X6F,0XBE,0XF7,0XEB,0X57,0XFD,0XF8,0X00,0X00,0X00,0X00,0X1F,
  0XFF,0XBF,0XD7,0X7E,0XF7,0X6F,0XAB,0XDE,0XBF,0XDF,0X7E,0XFD,0XBE,0XFD,0XF7,0XAD,
  0X2E,0XFF,0XDF,0XF0,0X00,0X00,0X00,0X00,0X1F,0XFF,0XEE,0XFB,0XEF,0XBB,0XFB,0XBF,
  0XFF,0XEB,0X7D,0XDF,0XDB,0X7B,0XD7,0X6D,0X72,0XFF,0XED,0XFF,0XE0,0X00,0X00,0X00,
  0X00,0X0F,0XFF,0XFD,0XEE,0XBB,0XEF,0XEF,0XDF,0XED,0XBF,0XFB,0XFE,0XED,0XFF,0XBD,
  0XF6,0XCF,0XDB,0XFF,0XBF,0XE0,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFD,0XFD,0X7D,
  0XFF,0X7B,0X7F,0XF7,0X6F,0XD7,0X77,0XED,0XD7,0X5B,0X37,0XFF,0XFF,0XFF,0XC0,0X00,
  0X00,0X00,0X00,0X03,0XFE,0XEF,0XDF,0X4E,0XEF,0XBB,0XBD,0XFD,0XDF,0XFF,0X7B,0XBF,
  0X76,0X7B,0XA9,0XDF,0X7E,0XB7,0XFF,0X80,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFB,
  0XF7,0XB5,0XEF,0XD7,0XB7,0X7D,0X55,0XDD,0X6B,0XDA,0XED,0X6A,0XFF,0XFF,0XFE,0XFF,
  0X80,0X00,0X00,0X00,0X00,0X01,0XFF,0XFE,0XEF,0XBA,0XDE,0XBB,0X7E,0XEE,0XEF,0XFF,
  0XEE,0XFD,0X77,0XAE,0XAF,0XFE,0XEF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,
  0XFF,0X7F,0XEF,0XB7,0XFF,0XBB,0XFF,0XB5,0X6E,0XB5,0XD7,0XAD,0X69,0X7F,0XEF,0XFF,
  0XDF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFD,0XF5,0XFA,0XAD,0XD7,0X55,
  0XDF,0XBA,0XBE,0XAA,0XDB,0XAB,0XDD,0X7F,0XFD,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,
  0X00,0X7F,0XFF,0XF7,0XBE,0XDF,0XEA,0XBA,0XF6,0XB5,0XD5,0X55,0X76,0XB5,0X5F,0XFF,
  0XFB,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XDF,0XFF,0X6A,0XBF,
  0X6F,0X5B,0XEE,0XB7,0XEE,0XDB,0X6A,0XAF,0XFF,0XFF,0XBF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X1F,0XFF,0XFF,0XD5,0XBD,0XAA,0XD2,0XED,0X55,0X5D,0X57,0X6A,0XAB,
  0XFE,0XEF,0XDF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFE,0XFF,0X7F,
  0XEF,0XFF,0XFF,0XBB,0X55,0XEB,0X6B,0XA5,0X2F,0X7F,0XFF,0XFF,0XFF,0XFB,0XE0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFB,0X7A,0XAB,0X6A,0XAC,0XAE,0XB5,0XD8,
  0X5A,0XFD,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,
  0XFF,0XEF,0XD7,0X6D,0X2A,0X53,0XFA,0XAD,0X27,0X57,0XFF,0XED,0XFF,0XFF,0XFB,0XBF,
  0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFB,0XFF,0XFB,0XB5,0XB7,0XDD,0X55,
  0XAA,0XDA,0XFF,0XFF,0XFF,0XDF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0XFF,0XFF,0XBF,0X7F,0XFE,0XDA,0XEE,0XAA,0XAB,0X7D,0XFB,0XFF,0XFF,0XFD,0XFB,
  0XEF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFD,0XFF,0X6B,0XFD,
  0XBB,0X55,0XFD,0XAF,0XDF,0XFB,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X3F,0XFE,0XFF,0XEF,0XFF,0X77,0XD5,0XFF,0X57,0XFD,0X7F,0XFF,0XEF,
  0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,
  0XDE,0XDE,0XBE,0XAB,0XFF,0X7F,0XFF,0XDF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XF7,0XFD,0XFF,0XF7,0XF7,0XFF,0XFF,0XF7,0XFF,
  0X7F,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,
  0XFF,0XF7,0XFF,0XDF,0X7D,0XFE,0XFD,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFE,0XFD,0XDF,0XDF,0XEF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X7B,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0XFF,0XFF,0XFF,0XF7,0XFE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XDF,0XBF,0XEF,0XFE,
  0XFF,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X3F,0XFB,0XFE,0XFF,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFD,
  0XFF,0XF7,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XF7,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFD,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFB,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XDD,0XBF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X7F,0XFF,0XFF,0XFB,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XBF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X0F,0XF7,0XFF,0XBF,0XFB,0XFF,0XFF,0XFF,0XFF,0XFF,0XFD,0XFF,0XC0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XF7,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XDF,0XFF,0XDF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0XF7,0XDF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFE,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFE,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XF7,0XFF,0XFF,0X7F,
  0XFD,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X1F,0XFF,0XEF,0XEB,0XFF,0XF7,0XFF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XBE,
  0XBF,0XFD,0XDF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XF0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00
};

// 200x200, 1 bit/pixel, row-major, MSB first; 1=white, 0=black.
const unsigned char gImage_eyelid_mask[5000] = {
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,
  0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,
  0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,
  0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XC0,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X0F,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,
  0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,
  0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X03,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,
  0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,
  0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XE0,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X1F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X3F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,
  0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFC,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFE,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,
  0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X01,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X01,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0X80,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0X80,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X07,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XC0,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X07,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,
  0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XE0,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X0F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XE0,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X0F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X0F,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,
  0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,
  0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0XE0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,
  0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,
  0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,
  0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,
  0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XF8,
  0X1F,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XF8,0X1F,0XFF,0XFF,0XFF,0XFC,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,
  0XFF,0XF8,0X1F,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XF8,0X1F,0XFF,0XFF,0XFF,0XFC,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,
  0XFF,0XFF,0XFF,0XF8,0X1F,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XF8,0X1F,0XFF,0XFF,
  0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X07,0XFF,0XFF,0XFF,0XF8,0X1F,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XF0,0X1F,
  0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,
  0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,
  0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XF0,0X1F,0XFF,0XFF,0XFF,
  0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0F,0XFF,0XFF,0XFF,0XF0,0X0F,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XF0,0X0F,0XFF,
  0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XF0,0X0F,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XF0,
  0X0F,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XE0,0X0F,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,
  0XFF,0XE0,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XE0,0X07,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,
  0XFF,0XFF,0XFF,0XE0,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XE0,0X07,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X07,0XFF,0XFF,0XFF,0XFF,0XC0,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XC0,0X07,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XC0,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,
  0XC0,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,
  0XFF,0XFF,0X80,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X01,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,
  0XFF,0XFF,0XFF,0XFF,0X80,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X01,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,
  0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,
  0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFE,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,
  0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X7F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFC,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X3F,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,
  0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,
  0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XE0,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X07,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X03,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,
  0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,
  0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFC,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X1F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,
  0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0X80,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,
  0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,
  0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X7F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFC,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X01,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X7F,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF8,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X1F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X07,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFE,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X3F,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XF0,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X07,0XFF,0XFF,
  0XFF,0XFF,0XFF,0XFF,0XFF,0XC0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X0F,0XFF,0XFF,0XFF,0XFF,0XFF,0XE0,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0XFF,0XFF,0XFF,0XFF,0XFE,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X03,0XFF,0XFF,0XFF,0X80,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,
  0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00
};

#endif // _YANKUANG_H_
```
