#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "audio_i2s.h"
#include "mfcc.h"
#include "main_functions.h"

#define FRAME_SIZE 512

// FIX #1: Lower VAD threshold
#define VAD_THRESHOLD 600

#define SILENCE_FRAMES_END 12

static int16_t audio_buffer[FRAME_SIZE];
static float audio_float[FRAME_SIZE];
static float mfcc[MFCC_COUNT];

static int speech_active = 0;
static int silence_count = 0;


// FIX #2: Add energy debug printing
static int detect_speech(int16_t *buffer)
{
    long energy = 0;

    for (int i = 0; i < FRAME_SIZE; i++)
        energy += abs(buffer[i]);

    energy /= FRAME_SIZE;

    // Debug print to tune threshold
    //printf("Energy: %ld\n", energy);

    static int speech_frames = 0;

    if (energy > VAD_THRESHOLD)
    {
        speech_frames++;
    }
    else
    {
        speech_frames = 0;
    }

    return speech_frames > 2;
}

void app_main(void)
{
    setup();
    audio_i2s_init();

    while (1)
    {
        audio_i2s_read(audio_buffer, FRAME_SIZE);

        int speech = detect_speech(audio_buffer);

        if (speech)
        {
            speech_active = 1;
            silence_count = 0;

            for (int i = 0; i < FRAME_SIZE; i++)
                audio_float[i] = (float)audio_buffer[i];

            mfcc_compute(audio_float, mfcc);

            loop(mfcc);
        }
        else
        {
            if (speech_active)
            {
                silence_count++;

                if (silence_count > SILENCE_FRAMES_END)
                {
                    // FIX #3: Debug print when inference starts
                    printf("Running inference...\n");

                    run_inference_on_speech();
                    reset_mfcc_buffer();

                    speech_active = 0;
                    silence_count = 0;
                }
            }
        }

        vTaskDelay(1);
    }
}