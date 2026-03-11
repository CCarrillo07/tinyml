#include <math.h>
#include <string.h>
#include "mfcc.h"
#include "esp_dsp.h"

#define SAMPLE_RATE 16000

static float window[FRAME_SIZE];
static float fft_buffer[FRAME_SIZE * 2];
static float mel_filterbank[MEL_FILTERS][FRAME_SIZE/2];

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void mel_filterbank_init()
{
    /* IMPORTANT FIX */
    memset(mel_filterbank, 0, sizeof(mel_filterbank));

    float mel_min = hz_to_mel(0);
    float mel_max = hz_to_mel(SAMPLE_RATE / 2);

    float mel_points[MEL_FILTERS + 2];
    float hz_points[MEL_FILTERS + 2];
    int bin[MEL_FILTERS + 2];

    for (int i = 0; i < MEL_FILTERS + 2; i++)
    {
        mel_points[i] = mel_min +
                        (mel_max - mel_min) * i / (MEL_FILTERS + 1);

        hz_points[i] = mel_to_hz(mel_points[i]);

        bin[i] = (int)((FRAME_SIZE + 1) *
                       hz_points[i] / SAMPLE_RATE);
    }

    for (int m = 1; m <= MEL_FILTERS; m++)
    {
        for (int k = bin[m-1]; k < bin[m]; k++)
        {
            mel_filterbank[m-1][k] =
                (float)(k - bin[m-1]) /
                (bin[m] - bin[m-1]);
        }

        for (int k = bin[m]; k < bin[m+1]; k++)
        {
            mel_filterbank[m-1][k] =
                (float)(bin[m+1] - k) /
                (bin[m+1] - bin[m]);
        }
    }
}

static int initialized = 0;

static void mfcc_init()
{
    if (initialized) return;

    dsps_wind_hann_f32(window, FRAME_SIZE);

    mel_filterbank_init();

    initialized = 1;
}

void mfcc_compute(int16_t *audio, float *mfcc_out)
{
    mfcc_init();

    for (int i = 0; i < FRAME_SIZE; i++)
    {
        fft_buffer[2*i] = (float)audio[i] * window[i];
        fft_buffer[2*i+1] = 0;
    }

    dsps_fft2r_fc32(fft_buffer, FRAME_SIZE);
    dsps_bit_rev_fc32(fft_buffer, FRAME_SIZE);
    dsps_cplx2reC_fc32(fft_buffer, FRAME_SIZE);

    float power[FRAME_SIZE/2];

    for (int i = 0; i < FRAME_SIZE/2; i++)
    {
        float real = fft_buffer[2*i];
        float imag = fft_buffer[2*i+1];

        power[i] = real*real + imag*imag;
    }

    float mel_energy[MEL_FILTERS] = {0};

    for (int m = 0; m < MEL_FILTERS; m++)
    {
        for (int k = 0; k < FRAME_SIZE/2; k++)
        {
            mel_energy[m] += power[k] * mel_filterbank[m][k];
        }
    }

    for (int i = 0; i < MEL_FILTERS; i++)
    {
        mel_energy[i] = logf(mel_energy[i] + 1e-10f);
    }

    dsps_dct_f32(mel_energy, MEL_FILTERS);

    for (int i = 0; i < MFCC_COUNT; i++)
    {
        mfcc_out[i] = mel_energy[i];
    }
}