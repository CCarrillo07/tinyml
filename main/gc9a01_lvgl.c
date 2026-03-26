#include "gc9a01_lvgl.h"
#include "lvgl.h"

#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h" 

#include "driver/spi_master.h"
#include "esp_timer.h"
#include <stdio.h>

#define SCREEN_WIDTH 240
#define SCREEN_HEIGHT 240

/* Pins */
#define PIN_SCLK 2
#define PIN_MOSI 3
#define PIN_DC   10
#define PIN_CS   7
#define PIN_RST  8

static lv_obj_t *label;
static lv_disp_draw_buf_t draw_buf;
static lv_color_t buf[SCREEN_WIDTH * 20];

static esp_lcd_panel_handle_t panel_handle;

/* Flush callback */
static void my_flush_cb(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *color_map)
{
    esp_lcd_panel_draw_bitmap(panel_handle,
        area->x1,
        area->y1,
        area->x2 + 1,
        area->y2 + 1,
        color_map);

    lv_disp_flush_ready(drv);
}

/* LVGL tick */
static void lv_tick_task(void *arg)
{
    lv_tick_inc(2);
}

/* Fade animation */
static void set_opa(void *obj, int32_t v)
{
    lv_obj_set_style_opa((lv_obj_t *)obj, v, 0);
}

void lcd_lvgl_init(void)
{
    /* SPI BUS */
    spi_bus_config_t buscfg = {
        .sclk_io_num = PIN_SCLK,
        .mosi_io_num = PIN_MOSI,
        .miso_io_num = -1,
        .max_transfer_sz = SCREEN_WIDTH * SCREEN_HEIGHT * 2
    };

    spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO);

    esp_lcd_panel_io_handle_t io_handle = NULL;

    /* ✅ FIX: use generic SPI config (correct for IDF 5.5) */
    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = PIN_DC,
        .cs_gpio_num = PIN_CS,
        .pclk_hz = 40 * 1000 * 1000,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .spi_mode = 0,
        .trans_queue_depth = 10,
    };

    esp_lcd_new_panel_io_spi(SPI2_HOST, &io_config, &io_handle);

    /* Panel config */
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = PIN_RST,
        .color_space = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
    };

    /* ✅ Keep ST7789 (compatible fallback for many GC9A01 boards) */
    esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle);

    esp_lcd_panel_reset(panel_handle);
    esp_lcd_panel_init(panel_handle);
    esp_lcd_panel_invert_color(panel_handle, true);
    esp_lcd_panel_disp_on_off(panel_handle, true);

    /* LVGL INIT */
    lv_init();

    lv_disp_draw_buf_init(&draw_buf, buf, NULL, SCREEN_WIDTH * 20);

    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);

    disp_drv.hor_res = SCREEN_WIDTH;
    disp_drv.ver_res = SCREEN_HEIGHT;
    disp_drv.flush_cb = my_flush_cb;
    disp_drv.draw_buf = &draw_buf;

    lv_disp_drv_register(&disp_drv);

    /* Tick timer */
    const esp_timer_create_args_t periodic_timer_args = {
        .callback = &lv_tick_task
    };

    esp_timer_handle_t timer;
    esp_timer_create(&periodic_timer_args, &timer);
    esp_timer_start_periodic(timer, 2000);

    /* Label */
    label = lv_label_create(lv_scr_act());
    lv_obj_set_style_text_font(label, &lv_font_montserrat_14, 0);
    lv_obj_center(label);
    lv_label_set_text(label, "...");
}

/* Display function */
void display_big_label(const char *text)
{
    lv_anim_t a;

    lv_anim_init(&a);
    lv_anim_set_var(&a, label);
    lv_anim_set_values(&a, LV_OPA_COVER, LV_OPA_TRANSP);
    lv_anim_set_time(&a, 300);
    lv_anim_set_exec_cb(&a, set_opa);
    lv_anim_start(&a);

    lv_timer_handler();

    lv_label_set_text(label, text);
    lv_obj_center(label);

    lv_anim_init(&a);
    lv_anim_set_var(&a, label);
    lv_anim_set_values(&a, LV_OPA_TRANSP, LV_OPA_COVER);
    lv_anim_set_time(&a, 300);
    lv_anim_set_exec_cb(&a, set_opa);
    lv_anim_start(&a);

    lv_timer_handler();
}