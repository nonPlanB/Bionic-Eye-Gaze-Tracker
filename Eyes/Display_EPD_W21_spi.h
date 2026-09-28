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
