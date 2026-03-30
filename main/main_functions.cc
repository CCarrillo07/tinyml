#include "main_functions.h"
#include "model.h"
#include "output_handler.h"
#include "mfcc.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <string.h>
#include "esp_log.h"

#include "gc9a01_lvgl.h"
#include "lvgl.h"

static const char* TAG = "TinyML";

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

/* ✅ Slightly larger arena (safe) */
constexpr int kTensorArenaSize = 70 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

/* ✅ MFCC buffer */
float mfcc_buffer[MAX_FRAMES][MFCC_COUNT];
int frame_index = 0;

/* ✅ NEW: prediction buffer (UI-safe) */
volatile int last_prediction = -1;
volatile bool new_prediction_available = false;

}

// =========================
void setup() {

    model = tflite::GetModel(g_model);

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddMaxPool2D();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddSoftmax();
    resolver.AddStridedSlice();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;

    lcd_lvgl_init();
    // 🔥 DEBUG: force text
    display_big_label("HELLO");
}

// =========================
void loop(float* mfcc) {

    if(frame_index < MAX_FRAMES){
        memcpy(mfcc_buffer[frame_index], mfcc, sizeof(float) * MFCC_COUNT);
        frame_index++;
    }
}

// =========================
extern "C" void run_inference_on_speech() {

    if(frame_index == 0) return;

    ESP_LOGI(TAG, "Frames: %d", frame_index);

    int offset = 0;

    for(int i = 0; i < MAX_FRAMES; i++) {

        for(int j = 0; j < MFCC_COUNT; j++) {

            float value = 0.0f;

            int src_index = i;

            if(src_index >= 0 && src_index < frame_index){
                value = mfcc_buffer[src_index][j];
            }

            int index = i * MFCC_COUNT + j;
            input->data.f[index] = value;
        }
    }

    if(interpreter->Invoke() == kTfLiteOk) {

        int count = output->dims->data[1];

        int max_idx = 0;
        float max_score = output->data.f[0];

        for(int i = 1; i < count; i++){
            if(output->data.f[i] > max_score){
                max_score = output->data.f[i];
                max_idx = i;
            }
        }

        ESP_LOGI(TAG, "Prediction: %s", kLabels[max_idx]);

        /* ✅ DO NOT CALL LVGL HERE */
        last_prediction = max_idx;
        new_prediction_available = true;
    }

    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

// =========================
extern "C" void reset_mfcc_buffer() {
    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

extern "C" int get_frame_count() {
    return frame_index;
}

/* =========================
   ✅ NEW: SAFE UI UPDATE FUNCTION
   ========================= */
extern "C" void update_display_if_needed() {

    if(new_prediction_available){

        new_prediction_available = false;

        if(last_prediction >= 0){
            display_big_label(kLabels[last_prediction]);
        }
    }
}