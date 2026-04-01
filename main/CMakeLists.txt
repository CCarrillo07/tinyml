idf_component_register(
    SRCS
        "audio_i2s.c"
        "mfcc.c"
        "main.c"
        "main_functions.cc"
        "constants.cc"
        "model.cc"
        "output_handler.cc"
        "gc9a01_lvgl_display.c"
    INCLUDE_DIRS "."
    REQUIRES driver freertos esp-dsp lvgl
)

set_source_files_properties(
    main_functions.cc constants.cc model.cc output_handler.cc
    PROPERTIES COMPILE_FLAGS "-std=c++17"
)

target_link_libraries(${COMPONENT_LIB} PUBLIC "-lstdc++")