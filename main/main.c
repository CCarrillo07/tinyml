#include <string.h>  
#include <math.h>    
#include "audio_i2s.h"
#include "mfcc.h"
#include "main_functions.h"
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lvgl.h"
#include "esp_log.h"

#define FRAME_SIZE 512
#define HOP_LENGTH 160

static const char *TAG = "VAD";

#define VAD_THRESHOLD 10000
#define VAD_MIN_ENERGY 800
#define SILENCE_FRAMES_END 15
#define MIN_SPEECH_FRAMES 20
#define SPEECH_START_FRAMES 3

static int16_t circular_buffer[FRAME_SIZE];
static float audio_float[FRAME_SIZE];
static float mfcc[MFCC_COUNT];

static int initialized = 0;
static int speech_active = 0;
static int silence_count = 0;

// =========================
// VAD function
// =========================
int detect_speech(int16_t *buffer){

    long energy = 0;
    for(int i = 0; i < FRAME_SIZE; i++){
        energy += abs(buffer[i]);
    }
    energy /= FRAME_SIZE;

    // ✅ DEBUG ONLY WHEN THRESHOLD EXCEEDED
    if(energy > VAD_THRESHOLD){
        ESP_LOGI(TAG, "Energy above threshold: %ld", energy);
    }

    if(energy < VAD_MIN_ENERGY){
        return 0;
    }

    static int speech_frames = 0;

    if(energy > VAD_THRESHOLD){
        speech_frames++;
    } else {
        speech_frames = 0;
    }

    return speech_frames > SPEECH_START_FRAMES;
}

// =========================
// Main loop
// =========================
void app_main(void){

    setup();
    audio_i2s_init();

    while(1){

        lv_timer_handler();

        memmove(circular_buffer,
                circular_buffer + HOP_LENGTH,
                (FRAME_SIZE - HOP_LENGTH) * sizeof(int16_t));

        audio_i2s_read(
            circular_buffer + (FRAME_SIZE - HOP_LENGTH),
            HOP_LENGTH
        );

        if(!initialized){
            initialized = 1;
            continue;
        }

        int speech = detect_speech(circular_buffer);

        if(speech){
            speech_active = 1;
            silence_count = 0;

            // Convert to float and normalize
            float max_val = 0.0f;
            for(int i=0;i<FRAME_SIZE;i++){
                audio_float[i] = (float)circular_buffer[i] / 32768.0f;
                if(fabsf(audio_float[i]) > max_val) max_val = fabsf(audio_float[i]);
            }
            float scale = (max_val>1e-6f)?max_val:1.0f;
            for(int i=0;i<FRAME_SIZE;i++) audio_float[i] /= scale;

            // Compute MFCC
            mfcc_compute(audio_float, mfcc);

            // 🔥 DEBUG: print first 5 MFCC coefficients
            ESP_LOGI(TAG, "MFCC[0..4]: %.3f %.3f %.3f %.3f %.3f",
                     mfcc[0], mfcc[1], mfcc[2], mfcc[3], mfcc[4]);

            loop(mfcc);

        } else {
            if(speech_active){
                silence_count++;
                if(silence_count > SILENCE_FRAMES_END){
                    int frames = get_frame_count();
                    ESP_LOGI(TAG, "Speech ended. Frames captured: %d", frames);
                    if(frames >= MIN_SPEECH_FRAMES){
                        ESP_LOGI(TAG, "Running inference...");
                        run_inference_on_speech();
                    } else {
                        ESP_LOGI(TAG, "Rejected (too short)");
                    }
                    reset_mfcc_buffer();
                    speech_active = 0;
                    silence_count = 0;
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}