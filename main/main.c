#include <string.h>
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

// NEW: Require minimum frames before inference
#define MIN_SPEECH_FRAMES 6

#define HOP_LENGTH 160    // <<< define hop length used in sliding buffer

static int16_t audio_buffer[FRAME_SIZE];
static float audio_float[FRAME_SIZE];
static float mfcc[MFCC_COUNT];

// Sliding buffer for MFCC computation
static float sliding_buffer[FRAME_SIZE] = {0};

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
        // Read new audio samples
        audio_i2s_read(audio_buffer, FRAME_SIZE);

        int speech = detect_speech(audio_buffer);

        if (speech)
        {
            speech_active = 1;
            silence_count = 0;

            // Convert int16 to float
            for (int i = 0; i < FRAME_SIZE; i++)
                audio_float[i] = (float)audio_buffer[i];

            // <<< UPDATED: slide buffer in HOP_LENGTH steps
            for (int i = 0; i <= FRAME_SIZE - HOP_LENGTH; i += HOP_LENGTH)
            {
                // Slide the buffer by HOP_LENGTH samples
                for (int j = 0; j < HOP_LENGTH; j++)
                {
                    memmove(sliding_buffer, sliding_buffer + 1, (FRAME_SIZE - 1) * sizeof(float));
                    sliding_buffer[FRAME_SIZE - 1] = audio_float[i + j];
                }

                // Compute MFCC for this hop
                mfcc_compute(sliding_buffer, mfcc);

                // Store MFCC frame
                loop(mfcc);
            }
        }
        else
        {
            if (speech_active)
            {
                silence_count++;

                // If enough silent frames, run inference
                if (silence_count > SILENCE_FRAMES_END)
                {
                    int frames = get_frame_count();

                    if (frames >= MIN_SPEECH_FRAMES)
                    {
                        printf("Running inference...\n");
                        run_inference_on_speech();
                    }
                    else
                    {
                        printf("Speech too short (%d frames)\n", frames);
                    }

                    reset_mfcc_buffer();
                    speech_active = 0;
                    silence_count = 0;
                }
            }
        }

        vTaskDelay(1);
    }
}