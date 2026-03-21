#include <SPI.h>
#include <RH_RF95.h>
#include <SD.h>
#include <Wire.h>
#include <Adafruit_LSM9DS1.h>
#include <Adafruit_Sensor.h> 
#include <TinyGPS++.h>
#include <Adafruit_AHRS.h>

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
float MX = 0;
float MY = 0;
float MZ = 0;
float velocity = 0;
const int chipSelect = 10;
File myFile;

Adafruit_Madgwick ahrs;       // from Adafruit_AHRS_Madgwick
float q0=1,q1=0,q2=0,q3=0;   // quaternion
float posX=0, posY=0, posZ=0;
float velX=0, velY=0, velZ=0;
unsigned long lastMicros = 0;
const float G = 9.80665f; // gravity (m/s^2)

const int CAL_SAMPLES = 2000;     // number of samples to use during calibration
const float ZUPT_ACC_THRESH = 0.2f;   // m/s^2 tolerance around gravity magnitude
const float ZUPT_GYRO_THRESH = 0.05f;  // rad/s tolerance for gyro (small)
const int ZUPT_WINDOW = 8;       // number of consecutive stationary checks required
int zuptCounter = 0;
// velocity bias estimator
float velBiasX = 0.0f, velBiasY = 0.0f, velBiasZ = 0.0f;
const float VEL_BIAS_ALPHA = 0.02f;    // how fast bias estimate adapts when stationary (0..1). 0.02 = slow
const float VEL_BIAS_DECAY = 0.999f;   // slight decay while moving

float prevWorldAx = 0, prevWorldAy = 0, prevWorldAz = 0; // for trapezoidal integration

// Optional bias estimates (tune after stationary calibration)
float accBiasX = 0, accBiasY = 0, accBiasZ = 0;
float gyroBiasX = 0, gyroBiasY = 0, gyroBiasZ = 0;

// Singleton instance of the radio driver
RH_RF95 rf95(RFM95_CS, RFM95_INT);

void setupSensor()
{
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G, lsm.LSM9DS1_ACCELDATARATE_50HZ);
  lsm.setupMag(lsm.LSM9DS1_MAGGAIN_4GAUSS);
  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);
}

// Helper: rotate vector by quaternion q (body->world): v' = q * v * q_conj
void rotateByQuat(float q0, float q1, float q2, float q3, float bx, float by, float bz,
                  float &wx, float &wy, float &wz) {
  // q * v (as quat with w=0)
  float ix =  q0*bx + q2*bz - q3*by;
  float iy =  q0*by + q3*bx - q1*bz;
  float iz =  q0*bz + q1*by - q2*bx;
  float iw = -q1*bx - q2*by - q3*bz;

  // result = ix,iy,iz * q_conj
  wx = ix*q0 - iw*q1 - iy*q3 + iz*q2;
  wy = iy*q0 - iw*q2 - iz*q1 + ix*q3;
  wz = iz*q0 - iw*q3 - ix*q2 + iy*q1;
}

// rotate a world vector (wx,wy,wz) into body frame using q (q0,q1,q2,q3)
void rotateWorldToBody(float q0, float q1, float q2, float q3,
                      float wx, float wy, float wz,
                      float &bx, float &by, float &bz) {
  // conjugate quaternion = (q0, -q1, -q2, -q3)
  // use rotateByQuat with conj q to rotate world->body
  rotateByQuat(q0, -q1, -q2, -q3, wx, wy, wz, bx, by, bz);
}

// NEW: Normalize quaternion to prevent drift
void normalizeQuaternion(float &q0, float &q1, float &q2, float &q3) {
  float norm = sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3);
  if (norm > 0.0001f) {  // avoid division by zero
    q0 /= norm;
    q1 /= norm;
    q2 /= norm;
    q3 /= norm;
  }
}

// ----------------- calibration routine (call once in setup) -----------------
void calibrateIMU() {
  Serial.println("Calibrating IMU... keep device perfectly still");
  if (myFile) { myFile.println("Calibrate start"); myFile.flush(); }

  // accumulators
  float sumGX=0, sumGY=0, sumGZ=0;
  float sumAX=0, sumAY=0, sumAZ=0;

  // IMPROVED: Increased warm-up from 50 to 200 for better AHRS convergence
  Serial.println("Warming up AHRS...");
  for (int i=0; i<500; i++) {
    sensors_event_t aev, mev, gev, tev;
    lsm.getEvent(&aev, &mev, &gev, &tev);
    // feed AHRS with what we have; small dt assumed
    ahrs.update(gev.gyro.x, gev.gyro.y, gev.gyro.z,
                aev.acceleration.x, aev.acceleration.y, aev.acceleration.z,
                mev.magnetic.x, mev.magnetic.y, mev.magnetic.z, 0.01f);
    delay(10);
  }

  Serial.println("Collecting calibration samples...");
  unsigned long start = micros();
  for (int i=0; i < CAL_SAMPLES; i++) {
    // read IMU
    sensors_event_t aev, mev, gev, tev;
    lsm.getEvent(&aev, &mev, &gev, &tev);

    // supply AHRS update with a measured small dt
    unsigned long now = micros();
    float dt = (now - lastMicros) * 1e-6f;
    if (dt <= 0 || dt > 0.05f) dt = 0.01f;
    lastMicros = now;

    ahrs.update(gev.gyro.x, gev.gyro.y, gev.gyro.z,
                aev.acceleration.x, aev.acceleration.y, aev.acceleration.z,
                mev.magnetic.x, mev.magnetic.y, mev.magnetic.z, dt);

    // accumulate raw gyro & accel (body frame)
    sumGX += gev.gyro.x;
    sumGY += gev.gyro.y;
    sumGZ += gev.gyro.z;

    sumAX += aev.acceleration.x;
    sumAY += aev.acceleration.y;
    sumAZ += aev.acceleration.z;

    delay( (int)(1000.0f / 100.0f) ); // aim for ~100Hz; adjust if your loop differs
  }

  // means
  float meanGX = sumGX / CAL_SAMPLES;
  float meanGY = sumGY / CAL_SAMPLES;
  float meanGZ = sumGZ / CAL_SAMPLES;

  float meanAX = sumAX / CAL_SAMPLES;
  float meanAY = sumAY / CAL_SAMPLES;
  float meanAZ = sumAZ / CAL_SAMPLES;

  // set gyro bias (rad/s)
  gyroBiasX = meanGX;
  gyroBiasY = meanGY;
  gyroBiasZ = meanGZ;

  // compute expected gravity in body frame using final quaternion from AHRS
  ahrs.getQuaternion(&q0, &q1, &q2, &q3);
  normalizeQuaternion(q0, q1, q2, q3);  // ADDED: Normalize after getting quaternion
  
  float gravBodyX, gravBodyY, gravBodyZ;
  // gravity in world is (0,0,G)
  rotateWorldToBody(q0, q1, q2, q3, 0.0f, 0.0f, G, gravBodyX, gravBodyY, gravBodyZ);

  // accelerometer bias = mean_accel - expected_gravity_body
  accBiasX = meanAX - gravBodyX;
  accBiasY = meanAY - gravBodyY;
  accBiasZ = meanAZ - gravBodyZ;

  Serial.print("Cal done. gyroBias (rad/s): ");
  Serial.print(gyroBiasX); Serial.print(", "); Serial.print(gyroBiasY); Serial.print(", "); Serial.println(gyroBiasZ);
  Serial.print("accBias (m/s^2): ");
  Serial.print(accBiasX); Serial.print(", "); Serial.print(accBiasY); Serial.print(", "); Serial.println(accBiasZ);
  if (myFile) {
    myFile.print("Cal gyroBias,"); myFile.print(gyroBiasX); myFile.print(","); myFile.print(gyroBiasY); myFile.print(","); myFile.println(gyroBiasZ);
    myFile.print("Cal accBias,"); myFile.print(accBiasX); myFile.print(","); myFile.print(accBiasY); myFile.print(","); myFile.println(accBiasZ);
    myFile.flush();
  }
  delay(200);
}

// Call once in setup:
void setupAHRS(float sampleHz) {
  ahrs = Adafruit_Madgwick();
  ahrs.begin(sampleHz); // set inv sample freq inside when using update(gx,..,dt) variant
  lastMicros = micros();
  
  // CRITICAL FIX: Increased beta from 0.01 to 0.1
  // Beta = 0.1 provides much faster correction while still being smooth
  // For rocket applications, this balances gyro precision with drift correction
  ahrs.setBeta(0.1f);
  
  Serial.print("AHRS initialized with beta = 0.1 at ");
  Serial.print(sampleHz);
  Serial.println(" Hz");
}

void setup() {
  digitalWrite(8, HIGH);
  Serial.begin(115200);  // Added Serial for debugging
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
    Serial.println("LoRa radio init failed");
    while (1);
  }
  Serial.println("LoRa radio init OK!");

  // Defaults after init are 434.0MHz, modulation GFSK_Rb250Fd250, +13dbM
  if (!rf95.setFrequency(RF95_FREQ)) {
    Serial.println("setFrequency failed");
    while (1);
  }
  Serial.print("Set Freq to: "); Serial.println(RF95_FREQ);

  // The default transmitter power is 13dBm, using PA_BOOST.
  // If you are using RFM95/96/97/98 modules which uses the PA_BOOST transmitter pin, then
  // you can set transmitter powers from 5 to 23 dBm:
  rf95.setTxPower(23, false);
  
  SD.remove("hell.txt");
  setupAHRS(100);
  myFile = SD.open("hell.txt",FILE_WRITE);
  if (!myFile) {
    Serial.println("Error opening hell.txt");
  } else {
    Serial.println("SD card file opened successfully");
  }
  
  calibrateIMU();
  Serial1.print("ya");
}

int16_t packetnum = 0;  // packet counter, we increment per xmission

// In your main loop, after you read sensor and have a, g, m:
void updatePosition_withCalibration(float ax, float ay, float az,
                                    float gx, float gy, float gz,
                                    float mx, float my, float mz) {
  unsigned long now = micros();
  float dt = (now - lastMicros) * 1e-6f;
  if (dt <= 0 || dt > 0.5f) dt = 0.01f;
  lastMicros = now;

  // remove biases
  ax -= accBiasX; ay -= accBiasY; az -= accBiasZ;
  gx -= gyroBiasX; gy -= gyroBiasY; gz -= gyroBiasZ;

  // update AHRS
  ahrs.update(gx, gy, gz, ax, ay, az, mx, my, mz, dt);
  ahrs.getQuaternion(&q0, &q1, &q2, &q3);
  
  // CRITICAL FIX: Normalize quaternion to prevent drift
  normalizeQuaternion(q0, q1, q2, q3);

  // rotate accel to world
  Serial.println("Quaternions");
  Serial.println(q0, 6);  // Increased precision for debugging
  Serial.println(q1, 6);
  Serial.println(q2, 6);
  Serial.println(q3, 6);
  
  // Verify quaternion is normalized (should be ~1.0)
  float qMag = sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3);
  Serial.print("Quat magnitude: "); Serial.println(qMag, 6);
  
  float wx, wy, wz;
  rotateByQuat(q0, q1, q2, q3, ax, ay, az, wx, wy, wz);

  // subtract gravity
  float axWorld = wx;
  float ayWorld = wy;
  float azWorld = wz - G;

  // ZUPT detection
  float accRes = sqrtf(axWorld*axWorld + ayWorld*ayWorld + azWorld*azWorld);
  float gyroMag = sqrtf(gx*gx + gy*gy + gz*gz);
  bool isStationary = (accRes < ZUPT_ACC_THRESH) && (gyroMag < ZUPT_GYRO_THRESH);

  if (isStationary) {
    zuptCounter++;
    if (zuptCounter >= ZUPT_WINDOW) {
      // If we have residual velocity, use it to update velBias (so future motion won't inherit this drift)
      // We update velocity bias with a small alpha so it converges slowly and doesn't overcorrect.
      velBiasX = (1.0f - VEL_BIAS_ALPHA) * velBiasX + VEL_BIAS_ALPHA * velX;
      velBiasY = (1.0f - VEL_BIAS_ALPHA) * velBiasY + VEL_BIAS_ALPHA * velY;
      velBiasZ = (1.0f - VEL_BIAS_ALPHA) * velBiasZ + VEL_BIAS_ALPHA * velZ;

      // Zero velocities now (ZUPT)
      velX = velY = velZ = 0.0f;

      // Optionally refine accelerometer bias a little using measured accel vs expected gravity.
      // Compute expected gravity in body frame and update accBias toward the difference (very slow).
      ahrs.getQuaternion(&q0, &q1, &q2, &q3); // ensure q's are current
      normalizeQuaternion(q0, q1, q2, q3);  // ADDED: Normalize here too
      
      float expGx, expGy, expGz;
      rotateWorldToBody(q0, q1, q2, q3, 0.0f, 0.0f, G, expGx, expGy, expGz);
      
      // measured mean accel in body (before subtracting accBias) approx = expG + accBias. So accBias ≈ meas - expG.
      // We don't have meas-mean here; use current ax/ay/az + accBias (we subtracted earlier), so reconstruct measured:
      float measAx = ax + accBiasX;
      float measAy = ay + accBiasY;
      float measAz = az + accBiasZ;
      // tiny adapt:
      const float ACC_BIAS_ALPHA = 0.001f; // very slow
      accBiasX = (1.0f - ACC_BIAS_ALPHA) * accBiasX + ACC_BIAS_ALPHA * (measAx - expGx);
      accBiasY = (1.0f - ACC_BIAS_ALPHA) * accBiasY + ACC_BIAS_ALPHA * (measAy - expGy);
      accBiasZ = (1.0f - ACC_BIAS_ALPHA) * accBiasZ + ACC_BIAS_ALPHA * (measAz - expGz);
    }
  } else {
    // moving
    zuptCounter = 0;
    // gently decay velBias so it doesn't accumulate forever if conditions change
    velBiasX *= VEL_BIAS_DECAY;
    velBiasY *= VEL_BIAS_DECAY;
    velBiasZ *= VEL_BIAS_DECAY;
  }

  // trapezoidal integration (use prevWorldAx/.. stored globally)
  float avgAx = 0.5f * (prevWorldAx + axWorld);
  float avgAy = 0.5f * (prevWorldAy + ayWorld);
  float avgAz = 0.5f * (prevWorldAz + azWorld);

  velX += avgAx * dt;
  velY += avgAy * dt;
  velZ += avgAz * dt;

  // subtract estimated velocity bias BEFORE damping/position integration
  velX -= velBiasX;
  velY -= velBiasY;
  velZ -= velBiasZ;

  // damping
  const float damping = 0.9995f;
  velX *= damping;
  velY *= damping;
  velZ *= damping;

  // update prev accel
  prevWorldAx = axWorld;
  prevWorldAy = ayWorld;
  prevWorldAz = azWorld;

  // integrate velocity -> position
  posX += velX * dt;
  posY += velY * dt;
  posZ += velZ * dt;

  // tiny deadband
  const float velDeadband = 0.002f;
  if (fabs(velX) < velDeadband) velX = 0;
  if (fabs(velY) < velDeadband) velY = 0;
  if (fabs(velZ) < velDeadband) velZ = 0;
}

void loop() {
  unsigned long loopStart = millis();
  
  // GPS reading (non-blocking)
  while(Serial1.available() > 0) {
    gps.encode(Serial1.read());  // Read GPS data and parse it
    // If new location data is available, print it  
    if (gps.location.isUpdated()) {
      if (myFile) {
        myFile.print("Latitude: "); myFile.println(gps.location.lat(), 6);
        myFile.print("Longitude: "); myFile.println(gps.location.lng(), 6);
        myFile.print("Number of Satellites: "); myFile.println(gps.satellites.value());
        myFile.print("Speed: "); myFile.println(gps.speed.mph());
        myFile.print("Altitude: "); myFile.println(gps.altitude.meters());
      }
      Serial.println(gps.speed.mph());
    }
  }
  
  // Read IMU
  lsm.read();
  sensors_event_t a, m, g, temp; 
  lsm.getEvent(&a, &m, &g, &temp); 
  
  AX = a.acceleration.x;
  AY = a.acceleration.y;
  AZ = a.acceleration.z;

  GX = g.gyro.x;
  GY = g.gyro.y;
  GZ = g.gyro.z;
  
  MX = m.magnetic.x;
  MY = m.magnetic.y;
  MZ = m.magnetic.z;

  updatePosition_withCalibration(AX, AY, AZ, GX, GY, GZ, MX, MY, MZ);

  // Log data with quaternions
  String line = String(millis()) + "," +
                  "Quat: " + String(q0,6) + "," + String(q1,6) + "," + String(q2,6) + "," + String(q3,6) + "," +
                  "Position: " + String(posX,3) + "," + String(posY,3) + "," + String(posZ,3) + "," +
                  "Velocity: " + String(velX,3) + "," + String(velY,3) + "," + String(velZ,3) + "," +
                  "Acceleration: " + String(AX,4) + "," + String(AY,4) + "," + String(AZ,4) + "," +
                  "Gyro: "+ String(GX,4) + "," + String(GY,4) + "," + String(GZ,4);
  
  if (myFile) {
    myFile.println(line);
    myFile.flush(); // ensure data is physically written; you can remove for speed
  }
  Serial.println(line);

  // Radio transmission
  char radiopacket[30] = "Accel: ";
  snprintf(radiopacket, sizeof(radiopacket), "Accel: %.2f, %.2f, %.2f", AX,AY,AZ);
  radiopacket[29] = 0;
  
  rf95.send((uint8_t *)radiopacket, strlen(radiopacket));
  rf95.waitPacketSent();
  
  // Now wait for a reply
  uint8_t buf[RH_RF95_MAX_MESSAGE_LEN];
  uint8_t len = sizeof(buf);

  if (rf95.waitAvailableTimeout(100)) {
    // Should be a reply message for us now
    if (rf95.recv(buf, &len)) {
      // Handle received data if needed
    } 
  }
  
  // IMPROVED: Maintain consistent loop timing (aim for ~100Hz = 10ms)
  unsigned long loopTime = millis() - loopStart;
  if (loopTime < 10) {
    delay(10 - loopTime);
  }
}