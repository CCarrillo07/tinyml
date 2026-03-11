idf_component_register(
    SRCS "audio_i2s.c"
    "constants.cc"
    "main_functions.cc"
    "main.c" 
    "mfcc.c" 
    "model.cc" 
    "output_handler.cc"
    INCLUDE_DIRS "."   
    PRIV_REQUIRES driver freertos esp-dsp
)