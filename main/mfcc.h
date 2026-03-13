#pragma once
#include <stdint.h>

#define FRAME_SIZE 512
#define HOP_LENGTH 160   // <<< NEW: match Python training
#define MFCC_COUNT 13
#define MEL_FILTERS 26
#define MAX_FRAMES 100

void mfcc_compute(float *audio, float *mfcc_out);