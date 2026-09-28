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