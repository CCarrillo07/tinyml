#include "gc9a01_lvgl_display.h"
#include "esp_log.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <string.h>

static const char *TAG = "LVGL_DISPLAY";

/* =========================
   PIN CONFIG
   ========================= */
#define PIN_NUM_MOSI    10
#define PIN_NUM_CLK     8
#define PIN_NUM_CS      7
#define PIN_NUM_DC      2

#define LCD_HOST SPI2_HOST

static spi_device_handle_t spi;

/* =========================
   LVGL
   ========================= */
static lv_obj_t *label = NULL;
static QueueHandle_t display_queue;

/* =========================
   QUEUE MESSAGE
   ========================= */
typedef struct {
    char text[32];
} display_msg_t;

/* =========================
   LOW LEVEL SPI
   ========================= */
static void lcd_cmd(uint8_t cmd)
{
    gpio_set_level(PIN_NUM_DC, 0);

    spi_transaction_t t = {
        .length = 8,
        .tx_buffer = &cmd
    };

    spi_device_polling_transmit(spi, &t);
}

static void lcd_data(const uint8_t *data, int len)
{
    if (len == 0) return;

    gpio_set_level(PIN_NUM_DC, 1);

    spi_transaction_t t = {
        .length = len * 8,
        .tx_buffer = data
    };

    spi_device_polling_transmit(spi, &t);
}

/* ========================= */
static void gc9a01_hw_init(void)
{
    gpio_set_direction(PIN_NUM_DC, GPIO_MODE_OUTPUT);

    spi_bus_config_t buscfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = -1,
        .sclk_io_num = PIN_NUM_CLK,
    };

    spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO);

    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = 20 * 1000 * 1000,
        .mode = 0,
        .spics_io_num = PIN_NUM_CS,
        .queue_size = 4,
    };

    spi_bus_add_device(LCD_HOST, &devcfg, &spi);

    lcd_cmd(0xEF);
    lcd_cmd(0xEB); lcd_data((uint8_t[]){0x14}, 1);
    lcd_cmd(0xFE);
    lcd_cmd(0xEF);

    lcd_cmd(0x36); lcd_data((uint8_t[]){0x48}, 1);
    lcd_cmd(0x3A); lcd_data((uint8_t[]){0x05}, 1);

    lcd_cmd(0x11);
    vTaskDelay(pdMS_TO_TICKS(120));
    lcd_cmd(0x29);
}

/* ========================= */
static void set_addr_window(int x1, int y1, int x2, int y2)
{
    uint8_t data[4];

    lcd_cmd(0x2A);
    data[0] = x1 >> 8; data[1] = x1 & 0xFF;
    data[2] = x2 >> 8; data[3] = x2 & 0xFF;
    lcd_data(data, 4);

    lcd_cmd(0x2B);
    data[0] = y1 >> 8; data[1] = y1 & 0xFF;
    data[2] = y2 >> 8; data[3] = y2 & 0xFF;
    lcd_data(data, 4);

    lcd_cmd(0x2C);
}

/* ========================= */
static void my_flush_cb(lv_disp_drv_t *drv,
                        const lv_area_t *area,
                        lv_color_t *color_map)
{
    int width = area->x2 - area->x1 + 1;
    int height = area->y2 - area->y1 + 1;

    set_addr_window(area->x1, area->y1, area->x2, area->y2);

    gpio_set_level(PIN_NUM_DC, 1);

    spi_transaction_t t = {
        .length = width * height * 16,
        .tx_buffer = color_map
    };

    spi_device_polling_transmit(spi, &t);

    lv_disp_flush_ready(drv);
}

/* =========================
   LVGL TASK (DEDICATED CORE)
   ========================= */
static void lvgl_task(void *arg)
{
    display_msg_t msg;

    while (1)
    {
        /* Handle incoming messages */
        if (xQueueReceive(display_queue, &msg, 0) == pdTRUE)
        {
            if (label)
            {
                lv_label_set_text(label, msg.text);
                lv_obj_center(label);
            }
        }

        lv_timer_handler();
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

/* ========================= */
void gc9a01_init(void)
{
    ESP_LOGI(TAG, "Init GC9A01 + LVGL");

    gc9a01_hw_init();
    lv_init();

    display_queue = xQueueCreate(5, sizeof(display_msg_t));

    static lv_disp_draw_buf_t draw_buf;
    static lv_color_t buf[240 * 10];

    lv_disp_draw_buf_init(&draw_buf, buf, NULL, 240 * 10);

    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);

    disp_drv.flush_cb = my_flush_cb;
    disp_drv.draw_buf = &draw_buf;
    disp_drv.hor_res = 240;
    disp_drv.ver_res = 240;

    lv_disp_drv_register(&disp_drv);

    label = lv_label_create(lv_scr_act());
    lv_label_set_text(label, "Ready");
    lv_obj_center(label);

 
    xTaskCreatePinnedToCore(
        lvgl_task,
        "lvgl",
        4096,
        NULL,
        1,
        NULL,
        0
    );
}

/* ========================= */
void display_send_text(const char *text)
{
    if (!display_queue) return;

    display_msg_t msg;
    strncpy(msg.text, text, sizeof(msg.text) - 1);
    msg.text[sizeof(msg.text) - 1] = '\0';

    xQueueSend(display_queue, &msg, 0);
}