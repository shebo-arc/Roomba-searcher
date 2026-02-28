#include <Wire.h>
#include "MPU6050.h"

#define SDA_PIN 15
#define SCL_PIN 7

#define SAMPLE_RATE_HZ 50
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_RATE_HZ)

MPU6050 mpu;

unsigned long lastSampleTime = 0;
unsigned long startTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // I2C init with explicit pins
  Wire.begin(SDA_PIN, SCL_PIN);

  //Serial.println("Initializing MPU6050...");
  mpu.initialize();

  if (!mpu.testConnection()) {
    //Serial.println("❌ MPU6050 NOT FOUND");
    while (1);
  }

  // ✅ SWITCHED RANGES (BEST FOR GESTURES)
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);   // ±4g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);   // ±500 °/s

  //Serial.println("✅ MPU6050 connected");
  //Serial.println("time_ms,ax,ay,az,gx,gy,gz,acc_mag,gyro_mag");

  startTime = millis();
}

void loop() {
  if (millis() - lastSampleTime >= SAMPLE_PERIOD_MS) {
    lastSampleTime = millis();

    int16_t axr, ayr, azr, gxr, gyr, gzr;
    mpu.getMotion6(&axr, &ayr, &azr, &gxr, &gyr, &gzr);

    // ✅ Correct scaling for ±4g and ±500 dps
    float ax = axr / 16384.0;
    float ay = ayr / 16384.0;
    float az = azr / 16384.0;

    float gx = gxr / 65.5;
    float gy = gyr / 65.5;
    float gz = gzr / 65.5;

    float acc_mag  = sqrt(ax * ax + ay * ay + az * az);
    float gyro_mag = sqrt(gx * gx + gy * gy + gz * gz);

    unsigned long t = millis() - startTime;

    //Serial.print(t); Serial.print(",");
    //Serial.print(ax, 4); Serial.print(",");
    Serial.print(ay, 4); Serial.print(",");
    Serial.print(az, 4); Serial.print(",");
    Serial.print(gx, 2); Serial.print(",");
    //Serial.print(gy, 2); Serial.print(",");
    Serial.println(gz, 2); //Serial.print(",");
    //Serial.print(acc_mag, 4); Serial.print(",");
    //Serial.println(gyro_mag, 4);
  }
}
