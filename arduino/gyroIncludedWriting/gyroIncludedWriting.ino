#include <SPI.h>
#include <RH_RF95.h>
#include <SD.h>
#include <Wire.h>
#include <Adafruit_LSM9DS1.h>
#include <Adafruit_Sensor.h> 
#include <TinyGPS++.h>

#define RFM95_CS    8
#define RFM95_INT   3
#define RFM95_RST   4

// Change to 434.0 or other frequency, must match RX's freq!
#define RF95_FREQ 915.0

// Create an instance of the GPS library
TinyGPSPlus gps;

// i2c
Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();

const float RAD_TO_DEGREE = 57.29577951308232f;
float AX = 0;
float AY = 0;
float AZ = 0;
float GX = 0;
float GY = 0;
float GZ = 0;
float velocity = 0;
const int chipSelect = 10;
File myFile;

// Singleton instance of the radio driver
RH_RF95 rf95(RFM95_CS, RFM95_INT);

void setupSensor()
{
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G, lsm.LSM9DS1_ACCELDATARATE_10HZ);
  lsm.setupMag(lsm.LSM9DS1_MAGGAIN_4GAUSS);
  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);
}

void setup() {
  digitalWrite(8, HIGH);
  Serial1.begin(9600);  
  lsm.begin();
  setupSensor(); // helper to just set the default scaling we want, see above!
  delay(100);
  SD.begin(chipSelect);
  delay(100);

  pinMode(RFM95_RST, OUTPUT); //set output
  digitalWrite(RFM95_RST, HIGH); //turn on
  delay(100);

  // manual reset
  digitalWrite(RFM95_RST, LOW);
  delay(10);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);
  while (!rf95.init()) {
    while (1);
  }

  // Defaults after init are 434.0MHz, modulation GFSK_Rb250Fd250, +13dbM
  if (!rf95.setFrequency(RF95_FREQ)) {
    while (1);
  }
  // Defaults after init are 434.0MHz, 13dBm, Bw = 125 kHz, Cr = 4/5, Sf = 128chips/symbol, CRC on

  // The default transmitter power is 13dBm, using PA_BOOST.
  // If you are using RFM95/96/97/98 modules which uses the PA_BOOST transmitter pin, then
  // you can set transmitter powers from 5 to 23 dBm:
  rf95.setTxPower(23, false);
  SD.remove("hell.txt");
  Serial1.print("ya");
}

int16_t packetnum = 0;  // packet counter, we increment per xmission

void loop() {
  myFile = SD.open("hell.txt",FILE_WRITE);

  while(Serial1.available() > 0) {
    gps.encode(Serial1.read());  // Read GPS data and parse it
    // If new location data is available, print it  
    if (gps.location.isUpdated()) {
      myFile.print("Latitude: "); myFile.println(gps.location.lat(), 6);
      myFile.print("Longitude: "); myFile.println(gps.location.lng(), 6);
      myFile.print("Number of Satellites: "); myFile.println(gps.satellites.value());
      myFile.print("Speed: "); myFile.println(gps.speed.mph());
      myFile.print("Altitude: "); myFile.println(gps.altitude.meters());
      //Serial.print("Altitude: "); Serial.print(gps.altitude.meters());
      Serial.println(gps.speed.mph());
    }
  }
  lsm.read();
  sensors_event_t a, m, g, temp; lsm.getEvent(&a, &m, &g, &temp); 
  float distance = a.distance; int time = a.timestamp; float altitude = a.altitude;
  AX = a.acceleration.x;
  AY = a.acceleration.y;
  AZ = a.acceleration.z;
  GX = g.gyro.x * RAD_TO_DEGREE;
  GY = g.gyro.y * RAD_TO_DEGREE;
  GZ = g.gyro.z * RAD_TO_DEGREE;
  

  if (myFile) {
    myFile.print(" Distance:   "); myFile.println(distance); 
    myFile.print(" Time:  "); myFile.println(time); 
    myFile.print(" Altitude:  "); myFile.println(altitude);
 
    myFile.print("AccelX: "); myFile.print(AX);
    myFile.print(" Y: "); myFile.print(AY);
    myFile.print(" Z: "); myFile.print(AZ);
    Serial.print(AX); Serial.print(" : ");Serial.print(AY);Serial.print(" : ");Serial.println(AZ);

    myFile.print(" GyroX: "); myFile.print(GX);
    myFile.print(" Y: "); myFile.print(GY);
    myFile.print(" Z: "); myFile.print(GZ);
    Serial.print(GX); Serial.print(" : ");Serial.print(GY);Serial.print(" : ");Serial.println(GZ);
  } 
  myFile.close();
  char radiopacket[30] = "Accel: ";
  snprintf(radiopacket, sizeof(radiopacket), "Accel: %.2f, %.2f, %.2f", AX,AY,AZ);
  //itoa(packetnum++, radiopacket+13, 10);
  radiopacket[29] = 0;
  delay(10);
  rf95.send((uint8_t *)radiopacket, 30);
  delay(10);
  rf95.waitPacketSent();
  // Now wait for a reply
  uint8_t buf[RH_RF95_MAX_MESSAGE_LEN];
  uint8_t len = sizeof(buf);

  if (rf95.waitAvailableTimeout(100)) {
    // Should be a reply message for us now
    if (rf95.recv(buf, &len)) {
    } 
  }
}