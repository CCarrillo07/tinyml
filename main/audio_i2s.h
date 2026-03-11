#pragma once
#include <stdint.h>

void audio_i2s_init(void);
void audio_i2s_read(int16_t *buffer, int samples);

#define SAMPLE_RATE 16000