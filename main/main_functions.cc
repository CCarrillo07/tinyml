#include "main_functions.h"
#include "model.h"
#include "output_handler.h"
#include "mfcc.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "TinyML";

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 40*1024;
uint8_t tensor_arena[kTensorArenaSize];

float mfcc_buffer[MAX_FRAMES][MFCC_COUNT];
int frame_index = 0;

}

void setup() {
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

    static tflite::MicroInterpreter static_interpreter(model,resolver,tensor_arena,kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);

    memset(mfcc_buffer,0,sizeof(mfcc_buffer));
    frame_index=0;
}

void loop(float* mfcc) {
    if(frame_index<MAX_FRAMES){
        memcpy(mfcc_buffer[frame_index], mfcc, sizeof(float)*MFCC_COUNT);
        frame_index++;
    }
}

extern "C" void run_inference_on_speech() {
    if(frame_index == 0) return;

    // Ensure MAX_FRAMES x MFCC_COUNT buffer is filled
    int offset = (MAX_FRAMES - frame_index) / 2;

    for(int i = 0; i < MAX_FRAMES; i++) {
        for(int j = 0; j < MFCC_COUNT; j++) {
            float value = 0.0f;
            int src = i - offset;
            if(src >= 0 && src < frame_index) value = mfcc_buffer[src][j];

            // Quantize using input tensor params
            int32_t scaled = (int32_t)(value / input->params.scale + input->params.zero_point);
            if(scaled < 0) scaled = 0;
            if(scaled > 255) scaled = 255;

            input->data.uint8[i * MFCC_COUNT + j] = (uint8_t)scaled;
        }
    }

    // Run inference
    if(interpreter->Invoke() == kTfLiteOk) {
        int count = output->dims->data[1];
        float scores[count];

        for(int i = 0; i < count; i++)
            scores[i] = (output->data.uint8[i] - output->params.zero_point) * output->params.scale;

        HandleOutput(scores, count);
    }

    // Reset buffer
    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

extern "C" void reset_mfcc_buffer() { memset(mfcc_buffer,0,sizeof(mfcc_buffer)); frame_index=0; }
extern "C" int get_frame_count() { return frame_index; }