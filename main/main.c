#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "audio_i2s.h"
#include "mfcc.h"
#include "main_functions.h"
#include "gc9a01_lvgl_display.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

#define FRAME_SIZE 512
#define HOP_LENGTH 160

static const char *TAG = "VAD";

/* =========================
   SYSTEM STATE MACHINE
   ========================= */
typedef enum {
    STATE_LISTENING,
    STATE_RECORDING,
    STATE_PROCESSING
} system_state_t;

static system_state_t state = STATE_LISTENING;

/* =========================
   PARAMETERS
   ========================= */
#define VAD_THRESHOLD 10000
#define VAD_MIN_ENERGY 800

#define SILENCE_FRAMES_END 20
#define SPEECH_START_FRAMES 1

#define PRE_SPEECH_SAMPLES (4000)
#define MAX_AUDIO_SAMPLES (16000)

/* =========================
   BUFFERS
   ========================= */
static int16_t circular_buffer[FRAME_SIZE];
static int16_t pre_buffer[PRE_SPEECH_SAMPLES];
static int16_t speech_buffer[MAX_AUDIO_SAMPLES];

static int pre_index = 0;
static int pre_filled = 0;
static int speech_index = 0;

static int initialized = 0;
static int speech_active = 0;
static int silence_count = 0;

/* =========================
   VAD
   ========================= */
int detect_speech(int16_t *buffer)
{
    long energy = 0;

    for (int i = 0; i < FRAME_SIZE; i++) {
        energy += abs(buffer[i]);
    }

    energy /= FRAME_SIZE;

    if (energy < VAD_MIN_ENERGY) {
        return 0;
    }

    static int speech_frames = 0;

    if (energy > VAD_THRESHOLD) {
        speech_frames++;
    } else {
        speech_frames = 0;
    }

    return speech_frames >= SPEECH_START_FRAMES;
}

/* ========================= */
void store_pre_buffer(int16_t *frame)
{
    for (int i = 0; i < HOP_LENGTH; i++) {
        pre_buffer[pre_index] = frame[i];
        pre_index = (pre_index + 1) % PRE_SPEECH_SAMPLES;

        if (pre_filled < PRE_SPEECH_SAMPLES) {
            pre_filled++;
        }
    }
}

void flush_pre_buffer()
{
    int start = (pre_index - pre_filled + PRE_SPEECH_SAMPLES) % PRE_SPEECH_SAMPLES;

    for (int i = 0; i < pre_filled; i++) {
        int idx = (start + i) % PRE_SPEECH_SAMPLES;

        if (speech_index < MAX_AUDIO_SAMPLES) {
            speech_buffer[speech_index++] = pre_buffer[idx];
        }
    }

    ESP_LOGI(TAG, "Injected RAW pre-buffer: %d samples", pre_filled);
}

/* ========================= */
void process_full_audio(int16_t *audio, int length)
{
    ESP_LOGI(TAG, "Generating MFCC sequence...");

    float frame[FRAME_SIZE];
    float mfcc[MFCC_COUNT];

    reset_mfcc_buffer();

    int frame_count = 0;
    float max_val = 1e-6f;

    for (int i = 0; i < length; i++) {
        float v = fabsf((float)audio[i]);
        if (v > max_val) max_val = v;
    }

    for (int i = 0; i < length - FRAME_SIZE; i += HOP_LENGTH) {

        for (int j = 0; j < FRAME_SIZE; j++) {
            frame[j] = audio[i + j] / max_val;
        }

        mfcc_compute(frame, mfcc);
        loop(mfcc);
        frame_count++;

        /* ✅ FEED WDT + YIELD */
        if (i % (HOP_LENGTH * 5) == 0) {
            esp_task_wdt_reset();
            taskYIELD();
        }
    }

    ESP_LOGI(TAG, "Total MFCC frames: %d", frame_count);

    /* ✅ FEED before heavy inference */
    esp_task_wdt_reset();

    run_inference_on_speech();

    /* ✅ FEED after inference */
    esp_task_wdt_reset();
}

/* ========================= */
void app_main(void)
{
    setup();
    audio_i2s_init();
    gc9a01_init();

    /* =========================
   LVGL Tick Function
   ========================= */
    void lvgl_tick_callback(void* arg) {
        lv_tick_inc(1);
    }

    /* =========================
    Timer creation
    ========================= */
    const esp_timer_create_args_t lvgl_tick_timer_args = {
        .callback = lvgl_tick_callback,
        .name = "lvgl_tick"
    };

    esp_timer_handle_t lvgl_tick_timer;
    esp_timer_create(&lvgl_tick_timer_args, &lvgl_tick_timer);
    esp_timer_start_periodic(lvgl_tick_timer, 1000);

    esp_task_wdt_add(NULL);

    display_send_text("Listening...");

    while (1)
    {
        display_task(); 
        if (state == STATE_PROCESSING) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        /* AUDIO PIPELINE */
        memmove(
            circular_buffer,
            circular_buffer + HOP_LENGTH,
            (FRAME_SIZE - HOP_LENGTH) * sizeof(int16_t)
        );

        audio_i2s_read(
            circular_buffer + (FRAME_SIZE - HOP_LENGTH),
            HOP_LENGTH
        );

        if (!initialized) {
            initialized = 1;
            continue;
        }

        store_pre_buffer(circular_buffer);

        int speech = detect_speech(circular_buffer);

        if (speech)
        {
            if (!speech_active) {
                ESP_LOGI(TAG, "Speech START detected");
                display_send_text("Recording...");

                speech_index = 0;
                flush_pre_buffer();

                state = STATE_RECORDING;
            }

            speech_active = 1;
            silence_count = 0;

            for (int i = 0; i < HOP_LENGTH; i++) {
                if (speech_index < MAX_AUDIO_SAMPLES) {
                    speech_buffer[speech_index++] = circular_buffer[i];
                }
            }
        }
        else
        {
            if (speech_active)
            {
                silence_count++;

                if (silence_count > SILENCE_FRAMES_END)
                {
                    ESP_LOGI(TAG, "Speech ended. Samples: %d", speech_index);

                    speech_active = 0;
                    silence_count = 0;

                    if (speech_index > 5000)
                    {
                        state = STATE_PROCESSING;
                        display_send_text("Processing...");

                        process_full_audio(speech_buffer, speech_index);

                        update_display_if_needed();
                    }

                    pre_filled = 0;
                    pre_index = 0;

                    state = STATE_LISTENING;
                    display_send_text("Listening...");
                }
            }
        }

        /* ✅ FEED WDT continuously */
        esp_task_wdt_reset();

        vTaskDelay(pdMS_TO_TICKS(5));
    }
}