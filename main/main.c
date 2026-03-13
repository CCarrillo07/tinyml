#include <string.h>    
#include "audio_i2s.h"
#include "mfcc.h"
#include "main_functions.h"
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define FRAME_SIZE 512
#define VAD_THRESHOLD 12000
#define SILENCE_FRAMES_END 12
#define MIN_SPEECH_FRAMES 6
#define HOP_LENGTH 160

static int16_t audio_buffer[FRAME_SIZE];
static float audio_float[FRAME_SIZE];
static float mfcc[MFCC_COUNT];
static float sliding_buffer[FRAME_SIZE]={0};
static int speech_active=0;
static int silence_count=0;

int detect_speech(int16_t *buffer){
    long energy=0;
    for(int i=0;i<FRAME_SIZE;i++) energy+=abs(buffer[i]);
    energy/=FRAME_SIZE;

    static int speech_frames=0;
    if(energy>VAD_THRESHOLD) speech_frames++;
    else speech_frames=0;
    return speech_frames>2;
}

void app_main(void){
    setup();
    audio_i2s_init();

    while(1){
        audio_i2s_read(audio_buffer, FRAME_SIZE);
        int speech=detect_speech(audio_buffer);

        if(speech){
            speech_active=1;
            silence_count=0;
            for(int i=0;i<FRAME_SIZE;i++) audio_float[i]=(float)audio_buffer[i];

            for(int i=0;i<=FRAME_SIZE-HOP_LENGTH;i+=HOP_LENGTH){
                for(int j=0;j<HOP_LENGTH;j++){
                    memmove(sliding_buffer,sliding_buffer+1,(FRAME_SIZE-1)*sizeof(float));
                    sliding_buffer[FRAME_SIZE-1]=audio_float[i+j];
                }
                mfcc_compute(sliding_buffer, mfcc);
                loop(mfcc);
            }
        } else {
            if(speech_active){
                silence_count++;
                if(silence_count>SILENCE_FRAMES_END){
                    int frames=get_frame_count();
                    if(frames>=MIN_SPEECH_FRAMES){
                        run_inference_on_speech();
                    }
                    reset_mfcc_buffer();
                    speech_active=0;
                    silence_count=0;
                }
            }
        }
        vTaskDelay(1);
    }
}