#include "audio_i2s.h"
#include "mfcc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "main_functions.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define VAD_THRESHOLD 1200
#define VAD_SILENCE_FRAMES 60

#define HOP_LENGTH 160

static int16_t audio_window[FRAME_SIZE];
static int16_t audio_buffer[HOP_LENGTH];

float mfcc[MFCC_COUNT];

static int silence_counter = 0;
static int recording = 0;

void app_main(void)
{
    audio_i2s_init();
    setup();

    memset(audio_window, 0, sizeof(audio_window));

    while (1)
    {
        audio_i2s_read(audio_buffer, HOP_LENGTH);

        int energy = 0;

        for (int i = 0; i < HOP_LENGTH; i++)
        {
            energy += abs(audio_buffer[i]);
        }

        energy /= HOP_LENGTH;

        if (!recording)
        {
            if (energy > VAD_THRESHOLD)
            {
                printf("\nSpeech detected\n");

                recording = 1;
                silence_counter = 0;
            }
        }
        else
        {
            /* sliding window */

            memmove(
                audio_window,
                audio_window + HOP_LENGTH,
                (FRAME_SIZE - HOP_LENGTH) * sizeof(int16_t)
            );

            memcpy(
                audio_window + (FRAME_SIZE - HOP_LENGTH),
                audio_buffer,
                HOP_LENGTH * sizeof(int16_t)
            );

            mfcc_compute(audio_window, mfcc);

            loop();

            if (energy < VAD_THRESHOLD)
                silence_counter++;
            else
                silence_counter = 0;

            if (silence_counter > VAD_SILENCE_FRAMES)
            {
                printf("Speech finished\n\n");

                recording = 0;
                silence_counter = 0;

                reset_mfcc_buffer();
            }
        }
    }
}