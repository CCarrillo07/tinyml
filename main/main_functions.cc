#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"
#include "mfcc.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <string.h>

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
    model = tflite::GetModel(g_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema %d not supported", model->version());
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddRelu();

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}

void loop()
{
    if (frame_index < MAX_FRAMES)
    {
        memcpy(
            mfcc_buffer[frame_index],
            mfcc,
            sizeof(float) * MFCC_COUNT
        );

        frame_index++;
    }

    if (frame_index >= MAX_FRAMES)
    {
        float mfcc_min = mfcc_buffer[0][0];
        float mfcc_max = mfcc_buffer[0][0];

        for (int i = 0; i < MAX_FRAMES; i++)
        {
            for (int j = 0; j < MFCC_COUNT; j++)
            {
                if (mfcc_buffer[i][j] < mfcc_min)
                    mfcc_min = mfcc_buffer[i][j];

                if (mfcc_buffer[i][j] > mfcc_max)
                    mfcc_max = mfcc_buffer[i][j];
            }
        }

        float range = mfcc_max - mfcc_min;
        if (range < 1e-6f) range = 1.0f;

        for (int i = 0; i < MAX_FRAMES; i++)
        {
            for (int j = 0; j < MFCC_COUNT; j++)
            {
                float normalized =
                    (mfcc_buffer[i][j] - mfcc_min) / range * 255.0f;

                int32_t scaled =
                    (int32_t)(normalized / input->params.scale +
                              input->params.zero_point);

                if (scaled < 0) scaled = 0;
                if (scaled > 255) scaled = 255;

                input->data.uint8[i * MFCC_COUNT + j] =
                    (uint8_t)scaled;
            }
        }

        if (interpreter->Invoke() == kTfLiteOk)
        {
            int count = output->dims->data[1];
            float scores[count];

            for (int i = 0; i < count; i++)
            {
                scores[i] =
                    (output->data.uint8[i] -
                     output->params.zero_point) *
                    output->params.scale;
            }

            HandleOutput(scores, count);
        }

        // only ONE inference
        memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
        frame_index = 0;
    }
}

extern "C" void reset_mfcc_buffer()
{
    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    frame_index = 0;
}