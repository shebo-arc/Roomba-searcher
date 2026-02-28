// ================================================================
// model_params.h  —  AUTO-GENERATED, DO NOT EDIT
// Declarations only. Data lives in model_params.cpp.
// This pattern avoids duplicate-symbol bugs on ESP32/Arduino.
// ================================================================
#pragma once
#include <stdint.h>

#define N_SUPPORT   361
#define N_FEATURES  28
#define N_CLASSES   5
#define N_PAIRS     10
#define GAMMA       0.1000000000f

extern const char* const CLASS_NAMES[N_CLASSES];

extern const float   SC_MEAN[N_FEATURES];
extern const float   SC_SCALE[N_FEATURES];
extern const float   SUPPORT_VECTORS[N_SUPPORT * N_FEATURES];
extern const float   PAIR_COEF[N_PAIRS * N_SUPPORT];
extern const float   INTERCEPT[N_PAIRS];
extern const float   PROB_A[N_PAIRS];
extern const float   PROB_B[N_PAIRS];
extern const uint8_t PAIR_I[N_PAIRS];
extern const uint8_t PAIR_J[N_PAIRS];
