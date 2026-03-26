idf_component_register(
    SRCS "gc9a01_lvgl.c"
         "audio_i2s.c"
         "constants.cc"
         "main_functions.cc"
         "main.c"
         "mfcc.c"
         "model.cc"
         "output_handler.cc"
    INCLUDE_DIRS "."
    REQUIRES driver freertos esp-dsp lvgl esp_lcd
)