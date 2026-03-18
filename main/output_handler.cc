#include <stdio.h>
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_log.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* kLabels[] = {
    "yes","no","on","off","up","down","left","right","stop","go"
};

void HandleOutput(float* scores, int count) {

    MicroPrintf("---- Prediction ----");

    for (int i = 0; i < count; i++) {
        MicroPrintf("%s: %.2f", kLabels[i], scores[i]);
    }

    MicroPrintf("-------------------");
}

#ifdef __cplusplus
}
#endif