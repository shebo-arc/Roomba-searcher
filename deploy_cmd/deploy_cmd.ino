#include <Wire.h>
#include "MPU6050.h"
#include <math.h>
#include "model_params.h"
#include <esp_now.h>
#include <WiFi.h>

#define SDA_PIN 15
#define SCL_PIN 7

#define SAMPLE_RATE_HZ    50
#define SAMPLE_PERIOD_MS  (1000 / SAMPLE_RATE_HZ)

#define WINDOW_SIZE        75
#define N_CHANNELS         4
#define VOTE_BUFFER_SIZE   5
#define MIN_VOTE_COUNT     3
#define BAUD_RATE          115200

MPU6050 mpu;

float buf[WINDOW_SIZE][N_CHANNELS];
int   buf_head  = 0;
int   buf_count = 0;

unsigned long lastSampleTime = 0;
int           last_printed   = -1;

#define HOP_SIZE  25
int hop_counter = 0;

int  vote_buf[VOTE_BUFFER_SIZE];
int  vote_pos      = 0;
bool vote_buf_full = false;

static float* K_cache = nullptr;

// ── ESP-NOW ───────────────────────────────────────────────────
// Replace with your receiver's MAC address
uint8_t receiverMAC[] = {0x84, 0x1F, 0xE8, 0x2B, 0x37, 0x8C};

// Legacy send callback (required signature for ESP-IDF legacy API)
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    Serial.print("ESP-NOW send: ");
    Serial.println(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
}

// Map class index → gesture code 1-5
// Adjust CLASS_NAMES order to match your model's indices
uint8_t gestureCode(int classIdx) {
    const char* name = CLASS_NAMES[classIdx];
    if (strcmp(name, "forward") == 0) return 1;
    if (strcmp(name, "left")    == 0) return 2;
    if (strcmp(name, "right")   == 0) return 3;
    if (strcmp(name, "idle")    == 0) return 4;
    if (strcmp(name, "stop")    == 0) return 5;
    return 0;   // unknown
}
// ─────────────────────────────────────────────────────────────

void extract_features(float feat[N_FEATURES]) {
    int fi = 0;
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        float sig[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++)
            sig[i] = buf[(buf_head + i) % WINDOW_SIZE][ch];

        float mean = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) mean += sig[i];
        mean /= WINDOW_SIZE;

        float mn = sig[0], mx = sig[0];
        for (int i = 1; i < WINDOW_SIZE; i++) {
            if (sig[i] < mn) mn = sig[i];
            if (sig[i] > mx) mx = sig[i];
        }

        float sum_sq = 0, m2 = 0, m3 = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float v = sig[i], d = v - mean, d2 = d * d;
            sum_sq += v * v;
            m2 += d2;
            m3 += d2 * d;
        }
        m2 /= WINDOW_SIZE;
        m3 /= WINDOW_SIZE;
        float std_v  = sqrtf(m2);
        float rms    = sqrtf(sum_sq / WINDOW_SIZE);
        float sk     = (std_v < 1e-9f) ? 0.0f : (m3 / (std_v * std_v * std_v));
        float energy = sum_sq / WINDOW_SIZE;

        int zc = 0;
        for (int i = 0; i < WINDOW_SIZE - 1; i++)
            if (sig[i] * sig[i + 1] < 0.0f) zc++;

        feat[fi++] = mn;
        feat[fi++] = mx;
        feat[fi++] = mx - mn;
        feat[fi++] = rms;
        feat[fi++] = sk;
        feat[fi++] = energy;
        feat[fi++] = (float)zc / WINDOW_SIZE;
    }
}

void scale_features(float feat[N_FEATURES]) {
    for (int i = 0; i < N_FEATURES; i++)
        feat[i] = (feat[i] - SC_MEAN[i]) / SC_SCALE[i];
}

void compute_kernel_cache(const float x[N_FEATURES]) {
    for (int i = 0; i < N_SUPPORT; i++) {
        float dist2 = 0.0f;
        int   base  = i * N_FEATURES;
        for (int j = 0; j < N_FEATURES; j++) {
            float d = x[j] - SUPPORT_VECTORS[base + j];
            dist2 += d * d;
        }
        K_cache[i] = expf(-GAMMA * dist2);
    }
}

void svm_decision(float scores[N_PAIRS]) {
    for (int p = 0; p < N_PAIRS; p++) {
        float s   = INTERCEPT[p];
        int   base = p * N_SUPPORT;
        for (int k = 0; k < N_SUPPORT; k++)
            s += PAIR_COEF[base + k] * K_cache[k];
        scores[p] = s;
    }
}

int ovo_vote(float scores[N_PAIRS]) {
    int votes[N_CLASSES] = {0};
    for (int p = 0; p < N_PAIRS; p++) {
        if (scores[p] > 0) votes[PAIR_I[p]]++;
        else                votes[PAIR_J[p]]++;
    }
    int best = 0;
    for (int i = 1; i < N_CLASSES; i++)
        if (votes[i] > votes[best]) best = i;
    return best;
}

int classify() {
    float feat[N_FEATURES];
    extract_features(feat);
    scale_features(feat);
    compute_kernel_cache(feat);
    float scores[N_PAIRS];
    svm_decision(scores);
    return ovo_vote(scores);
}

int push_vote(int pred) {
    vote_buf[vote_pos] = pred;
    vote_pos = (vote_pos + 1) % VOTE_BUFFER_SIZE;
    if (vote_pos == 0) vote_buf_full = true;
    if (!vote_buf_full) return -1;

    int counts[N_CLASSES] = {0};
    for (int i = 0; i < VOTE_BUFFER_SIZE; i++)
        counts[vote_buf[i]]++;

    int winner = 0;
    for (int i = 1; i < N_CLASSES; i++)
        if (counts[i] > counts[winner]) winner = i;

    return (counts[winner] >= MIN_VOTE_COUNT) ? winner : -1;
}

void setup() {
    Serial.begin(BAUD_RATE);
    delay(1000);

    K_cache = (float*) malloc(N_SUPPORT * sizeof(float));
    if (!K_cache) {
        Serial.println("malloc failed for K_cache!");
        while (1);
    }

    Wire.begin(SDA_PIN, SCL_PIN);
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("MPU6050 not found!");
        while (1);
    }
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);

    // ── ESP-NOW init ──────────────────────────────────────────
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed!");
        while (1);
    }

    // Register legacy send callback
    esp_now_register_send_cb(onDataSent);

    // Register peer
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, receiverMAC, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add ESP-NOW peer!");
        while (1);
    }
    // ─────────────────────────────────────────────────────────
}

void loop() {
    if (millis() - lastSampleTime < SAMPLE_PERIOD_MS) return;
    lastSampleTime = millis();

    int16_t axr, ayr, azr, gxr, gyr, gzr;
    mpu.getMotion6(&axr, &ayr, &azr, &gxr, &gyr, &gzr);

    float ay = ayr / 16384.0f;
    float az = azr / 16384.0f;
    float gx = gxr / 65.5f;
    float gz = gzr / 65.5f;

    buf[buf_head][0] = ay;
    buf[buf_head][1] = az;
    buf[buf_head][2] = gx;
    buf[buf_head][3] = gz;
    buf_head = (buf_head + 1) % WINDOW_SIZE;
    if (buf_count < WINDOW_SIZE) buf_count++;
    if (buf_count < WINDOW_SIZE) return;

    hop_counter++;
    if (hop_counter < HOP_SIZE) return;
    hop_counter = 0;

    unsigned long t0 = micros();
    int pred = classify();
    unsigned long ms = (micros() - t0) / 1000UL;

    int majority = push_vote(pred);
    if (majority == -1) return;
    if (majority == last_printed) return;

    int counts[N_CLASSES] = {0};
    for (int i = 0; i < VOTE_BUFFER_SIZE; i++)
        counts[vote_buf[i]]++;

    Serial.print("Gesture: ");
    Serial.print(CLASS_NAMES[majority]);
    Serial.print("   (");
    for (int i = 0; i < N_CLASSES; i++) {
        if (counts[i] > 0) {
            Serial.print(CLASS_NAMES[i]);
            Serial.print(":");
            Serial.print(counts[i]);
            Serial.print(" ");
        }
    }
    Serial.print(")  ");
    Serial.print(ms);
    Serial.println("ms");

    // ── Send gesture code via ESP-NOW ─────────────────────────
    uint8_t code = gestureCode(majority);
    if (code != 0) {
        esp_now_send(receiverMAC, &code, sizeof(code));
    }
    // ─────────────────────────────────────────────────────────

    last_printed = majority;
}