#include <SPI.h>
#include <RH_RF95.h>
#include <SD.h>
#include <Wire.h>
#include <Adafruit_LSM9DS1.h>
#include <Adafruit_Sensor.h> 
#include <TinyGPS++.h>

// GPS and Sensors
TinyGPSPlus gps;
Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();

// SD card
const int chipSelect = 10;
File myFile;

// Timing
unsigned long lastLogTime = 0;
const unsigned long logInterval = 200;  // 200ms between sensor/log writes
unsigned long lastFlushTime = 0;
const unsigned long flushInterval = 5000; // flush SD every 5s

void setupSensor() {
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G, lsm.LSM9DS1_ACCELDATARATE_10HZ);
  lsm.setupMag(lsm.LSM9DS1_MAGGAIN_4GAUSS);
  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);
}

void setup() {
  digitalWrite(8, HIGH);  
  Serial1.begin(9600);  
  lsm.begin();
  setupSensor();
  SD.begin(chipSelect);
  SD.remove("data.txt");
  myFile = SD.open("data.txt", FILE_WRITE);
}

void loop() {
  while (Serial1.available() > 0) {
    gps.encode(Serial1.read());
  }
  unsigned long currentMillis = millis();


  if (currentMillis - lastLogTime >= logInterval) {
    lastLogTime = currentMillis;

    // Read sensor data
    sensors_event_t a, m, g, temp;
    lsm.getEvent(&a, &m, &g, &temp);
    float AX = a.acceleration.x;
    float AY = a.acceleration.y;
    float AZ = a.acceleration.z;
    float Rx = g.gyro.x;
    float Ry = g.gyro.y;
    float Rz = g.gyro.z;
    float distance = a.distance;
    int timeStamp = a.timestamp;
    float altitude = a.altitude;


    // Prepare GPS data
    double lat = gps.location.isUpdated() ? gps.location.lat() : 0.0;
    double lng = gps.location.isUpdated() ? gps.location.lng() : 0.0;
    int sats = gps.satellites.isUpdated() ? gps.satellites.value() : 0;
    double speed = gps.speed.isUpdated() ? gps.speed.mph() : 0.0;
    double gpsAlt = gps.altitude.isUpdated() ? gps.altitude.meters() : 0.0;

    // Write to SD card in a single operation
    if (myFile) {
      char buf[256];
      int n = snprintf(buf, sizeof(buf),
                       "Lat: %.6f, Lon: %.6f, Sats: %d, Speed: %.2f mph, GPS Alt: %.2f m\n"
                       "Distance: %.2f, Time: %d, Altitude: %.2f\n"
                       "Accel X: %.2f m/s^2, Y: %.2f m/s^2, Z: %.2f m/s^2\n\n",
                       lat, lng, sats, speed, gpsAlt,
                       distance, timeStamp, altitude,
                       AX, AY, AZ);
      myFile.write(buf, n);
    }

    // Optional debug print
    Serial.print(F("AX: ")); Serial.println(AX);
  }

  // -------------------------
  // 3. Periodic SD flush to reduce wear
  // -------------------------
  if (currentMillis - lastFlushTime >= flushInterval) {
    myFile.flush();
    lastFlushTime = currentMillis;
  }
}
