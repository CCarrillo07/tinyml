#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"
#include "mfcc.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_log.h"
#include <string.h>
#include <math.h>

static const char* TAG = "TinyML";

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 40 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

float mfcc_buffer[MAX_FRAMES][MFCC_COUNT];
int frame_index = 0;

}

void setup()
{
    esp_log_level_set("*", ESP_LOG_INFO);

    model = tflite::GetModel(g_model);

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddMaxPool2D();
    resolver.AddPack();
    resolver.AddQuantize();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddSoftmax();
    resolver.AddStridedSlice();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );

    interpreter = &static_interpreter;

    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "===== MODEL INFO =====");
    ESP_LOGI(TAG, "Input scale: %f", input->params.scale);
    ESP_LOGI(TAG, "Input zero_point: %d", input->params.zero_point);
    ESP_LOGI(TAG, "Output scale: %f", output->params.scale);
    ESP_LOGI(TAG, "Output zero_point: %d", output->params.zero_point);
    ESP_LOGI(TAG, "Input tensor bytes: %d", input->bytes);
    ESP_LOGI(TAG, "======================");

    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

void loop(float* mfcc)
{
    if (frame_index < MAX_FRAMES)
    {
        memcpy(mfcc_buffer[frame_index], mfcc, sizeof(float) * MFCC_COUNT);
        frame_index++;
    }
}

extern "C" void run_inference_on_speech()
{
    if (frame_index == 0) return;

    ESP_LOGI(TAG, "===== MFCC DEBUG =====");
    ESP_LOGI(TAG, "Frames captured: %d", frame_index);

    ESP_LOGI(TAG, "First MFCC frame:");
    for (int j = 0; j < MFCC_COUNT; j++)
        printf("%.2f ", mfcc_buffer[0][j]);
    printf("\n");

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int i = 0; i < frame_index; i++)
        for (int j = 0; j < MFCC_COUNT; j++)
        {
            sum += mfcc_buffer[i][j];
            sq_sum += mfcc_buffer[i][j] * mfcc_buffer[i][j];
        }

    float mean = sum / (frame_index * MFCC_COUNT);
    float std = sqrtf((sq_sum / (frame_index * MFCC_COUNT)) - (mean * mean));

    if (std < 1e-6f) std = 1e-6f;

    ESP_LOGI(TAG, "Mean: %f", mean);
    ESP_LOGI(TAG, "Std: %f", std);

    int offset = (MAX_FRAMES - frame_index) / 2;

    ESP_LOGI(TAG, "===== NORMALIZED MFCC =====");

    for (int j = 0; j < MFCC_COUNT; j++)
    {
        float v = (mfcc_buffer[0][j] - mean) / std;
        printf("%.2f ", v);
    }
    printf("\n");

    ESP_LOGI(TAG, "===== QUANTIZED INPUT =====");

    for (int i = 0; i < MAX_FRAMES; i++)
    for (int j = 0; j < MFCC_COUNT; j++)
    {
        float value = 0.0f;

        int src = i - offset;

        if (src >= 0 && src < frame_index)
            value = mfcc_buffer[src][j];

        float normalized = (value - mean) / std;

        int32_t scaled =
            (int32_t)(normalized / input->params.scale + input->params.zero_point);

        if (scaled < 0) scaled = 0;
        if (scaled > 255) scaled = 255;

        input->data.uint8[i * MFCC_COUNT + j] = (uint8_t)scaled;

        if (i == offset && j < MFCC_COUNT)
            printf("%d ", (int)scaled);
    }

    printf("\n");

    ESP_LOGI(TAG, "============================");

    if (interpreter->Invoke() == kTfLiteOk)
    {
        int count = output->dims->data[1];
        float scores[count];

        for (int i = 0; i < count; i++)
        {
            scores[i] =
                (output->data.uint8[i] - output->params.zero_point)
                * output->params.scale;
        }

        HandleOutput(scores, count);
    }

    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

extern "C" void reset_mfcc_buffer()
{
    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

// NEW: allow main.c to read frame count
extern "C" int get_frame_count()
{
    return frame_index;
}