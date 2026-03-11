#pragma once
#include <stdint.h>

#define FRAME_SIZE 512
#define MFCC_COUNT 13
#define MEL_FILTERS 26

/* Match training */
#define MAX_FRAMES 49

extern float mfcc[MFCC_COUNT];

void mfcc_compute(int16_t *audio, float *mfcc_out);