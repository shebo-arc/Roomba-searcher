#include <esp_now.h>
#include <WiFi.h>

// Motor pins
#define IN1 26
#define IN2 27
#define IN3 14
#define IN4 12

#define ENA 25
#define ENB 33

#define PWM_FREQ 1000
#define PWM_RES  8
#define SPEED    200

void forward(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(0, speed);
  ledcWrite(1, speed);
}

void backward(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(0, speed);
  ledcWrite(1, speed);
}

void left(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(0, 0);
  ledcWrite(1, speed);
}

void right(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(0, speed);
  ledcWrite(1, 0);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(0, 0);
  ledcWrite(1, 0);
}

// Legacy ESP-NOW receive callback
void onDataRecv(const uint8_t *mac_addr, const uint8_t *data, int len) {
  if (len < 1) return;
  uint8_t code = data[0];

  Serial.print("Received code: ");
  Serial.println(code);

  switch (code) {
    case 1: forward(SPEED);  break;
    case 2: left(SPEED);     break;
    case 3: right(SPEED);    break;
    case 4: stopMotors();    break;
    case 5: backward(SPEED); break;
    default:
      Serial.println("Unknown code, stopping.");
      stopMotors();
      break;
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  ledcSetup(0, PWM_FREQ, PWM_RES);
  ledcSetup(1, PWM_FREQ, PWM_RES);
  ledcAttachPin(ENA, 0);
  ledcAttachPin(ENB, 1);

  stopMotors();

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    while (1);
  }

  esp_now_register_recv_cb(onDataRecv);

  Serial.println("Receiver ready.");
  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress());
}

void loop() {
}