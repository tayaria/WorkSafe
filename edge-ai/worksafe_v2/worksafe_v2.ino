/*
  WORKSAFE v2 - Firmware embarque ESP32
  Materiel : ESP32, 2 x MPU-6050 (tibia 0x68 / cuisse 0x69), LM35, moteur vibrant (+ transistor),
             buzzer actif, 3 LED RGB, batterie Li-Po + TP4056

  Fonctions :
   - Angle reel du genou (cuisse/tibia), vitesse angulaire, hyperextension
   - Chocs, foulees, cadence (MPU tibia)
   - Fatigue : hausse des chocs, baisse d'amplitude de flexion, baisse de cadence
   - Echauffement (LM35)
   - Instabilite laterale (rotation hors plan sagittal)
   - Decision : danger si 1 indicateur critique OU 2 indicateurs en attention
   - Alertes : LED RGB, moteur vibrant, buzzer
   - BLE (Nordic UART) : trame 1 Hz + commandes 'C' (calibration) et 'M' (mute)
   - Log CSV pour dataset IA

  Si le MPU cuisse (0x69) est absent : mode 1 capteur (angle = inclinaison du tibia).
  Aucune bibliotheque externe a installer.
*/
#include <Arduino.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ===================== OPTIONS =====================
#define USE_MOTOR         1   // moteur vibrant branche (transistor + 1k)
#define USE_RGB           1   // LED RGB branchees (resistance sur la patte commune)
#define LED_COMMON_ANODE  0   // 0 = cathode commune, 1 = anode commune
#define BUZZER_ON_WARN    0   // 1 = le buzzer sonne aussi au niveau ATTENTION
#define USE_BATT_SENSE    0   // 1 = diviseur 100k/100k sur GPIO35
#define KNEE_SIGN         1   // mettre -1 si l'angle diminue quand tu fléchis le genou
#define LOG_CSV           0   // 1 = log CSV sur Serial (dataset IA)
#define LOG_CSV_DIV       4   // 1 ligne toutes les 4 mesures (50 Hz)

// ===================== OPTIONS IA (edge-ai) =====================
// L'IA (autoencoder + CNN + fusion) tourne en MODE OMBRE : elle calcule et affiche, elle ne commande RIEN.
// Avec USE_AI 0, le sketch se comporte exactement comme worksafe_v2.ino d'origine.
#define USE_AI            1   // 1 = IA compilee et active
#define AI_SHADOW         1   // 1 = mode ombre (obligatoire pour l'instant)
#define AI_SEPARATE_TASK  1   // 1 = inference dans une tache FreeRTOS separee (la boucle n'attend jamais) ; 0 = dans loop()
#define AI_DECIM_MODE     1   // 1 = moyenne des 4 mesures a 200 Hz ; 0 = une mesure sur 4 (comme l'ancien LOG_CSV)
#define AI_PRINT_SERIAL   1   // 1 = ajoute ",A:score,Y:classe,Z:etat,I:ms" a la trame serie 1 Hz
#define AI_PRINT_BLE      0   // 1 = ajoute aussi ces champs a la trame BLE (l'appli mobile doit les accepter)
#define AI_AUTOTEST_BOOT  1   // 1 = auto-test au demarrage (et commande T) ; 0 = vecteurs de test non compiles
#if USE_AI && !AI_SHADOW
#error "actionnement par l'IA non valide : garder AI_SHADOW a 1"
#endif
#if USE_AI
#include "src/ia/ia_worksafe.h"
#endif

// ===================== BROCHES =====================
#define PIN_SDA     21
#define PIN_SCL     22
#define PIN_LM35    34
#define PIN_VBAT    35
#define PIN_MOTOR   25
#define PIN_BUZZER  26
// LED RGB : {R, G, B}   LED1 = niveau, LED2 = BLE, LED3 = systeme / foulees
const uint8_t LED_PINS[3][3] = {{13, 14, 27}, {16, 17, 18}, {19, 23, 4}};

// ===================== SEUILS (a calibrer par athlete) =====================
#define STEP_G           2.0f    // detection d'une foulee (g, tibia)
#define IMPACT_G         3.0f    // comptage des chocs
#define PEAK_WARN_G      5.0f
#define PEAK_DANGER_G    8.0f
#define ANGLE_WARN       120.0f  // flexion (deg)
#define ANGLE_DANGER     140.0f
#define HYPER_WARN       -5.0f   // hyperextension (deg)
#define HYPER_DANGER     -10.0f
#define VEL_WARN         700.0f  // vitesse angulaire du genou (deg/s)
#define VEL_DANGER       900.0f
#define LAT_WARN         300.0f  // rotation hors plan sagittal (deg/s)
#define LAT_DANGER       500.0f
#define DT_WARN          2.5f    // echauffement vs repos (C)
#define DT_DANGER        4.0f
#define BASE_STRIDES     100     // foulees pour fixer la reference de fatigue
#define FAT_PEAK_WARN    25.0f   // hausse des chocs (%)
#define FAT_PEAK_DANGER  50.0f
#define FAT_AMP_WARN     15.0f   // baisse d'amplitude de flexion (%)
#define FAT_AMP_DANGER   25.0f
#define FAT_CAD_WARN     10.0f   // baisse de cadence (%)
#define FAT_CAD_DANGER   20.0f

#define RAD2DEG 57.2958f

// ===================== IMU =====================
struct Imu {
  uint8_t addr; bool ok;
  float ax, ay, az, gx, gy, gz;
  float bx, by, bz;          // biais gyro
  float raw, ref, ang;       // angle brut, reference, angle relatif
};
Imu shank = {0x68, false};   // tibia  (AD0 -> GND)
Imu thigh = {0x69, false};   // cuisse (AD0 -> 3V3)

// (struct Hold declaree ici, avant toute fonction : exigence du pre-processeur Arduino)
struct Hold { uint8_t lvl; unsigned long until; };

void mpuWrite(uint8_t a, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(a); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}
bool mpuInit(Imu &m) {
  Wire.beginTransmission(m.addr);
  if (Wire.endTransmission() != 0) return false;
  mpuWrite(m.addr, 0x6B, 0x00);   // reveil
  mpuWrite(m.addr, 0x1A, 0x01);   // filtre ~184 Hz
  mpuWrite(m.addr, 0x19, 0x04);   // 1 kHz / 5 = 200 Hz
  mpuWrite(m.addr, 0x1B, 0x18);   // gyro  +-2000 deg/s -> 16.4 LSB
  mpuWrite(m.addr, 0x1C, 0x18);   // accel +-16 g       -> 2048 LSB/g
  return true;
}
bool mpuRead(Imu &m) {
  Wire.beginTransmission(m.addr); Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(m.addr, (uint8_t)14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) { uint8_t hi = Wire.read(); uint8_t lo = Wire.read(); v[i] = (hi << 8) | lo; }
  m.ax = v[0] / 2048.0f; m.ay = v[1] / 2048.0f; m.az = v[2] / 2048.0f;
  m.gx = v[4] / 16.4f - m.bx; m.gy = v[5] / 16.4f - m.by; m.gz = v[6] / 16.4f - m.bz;
  return true;
}
float wrap180(float a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }

// Filtre complementaire : rotation autour de l'axe Y du module (axe du genou)
void updateAngle(Imu &m, float dt) {
  float mag = sqrtf(m.ax * m.ax + m.ay * m.ay + m.az * m.az);
  float acc = atan2f(-m.ax, m.az) * RAD2DEG;
  m.raw += m.gy * dt;
  if (mag > 0.7f && mag < 1.3f) m.raw += 0.02f * wrap180(acc - m.raw);
  m.raw = wrap180(m.raw);
  m.ang = wrap180(m.raw - m.ref);
}

// ===================== ETAT =====================
float tempC = 30, baseTemp = 30;
float knee = 0, kneeVel = 0, latRate = 0, vbat = 0;
float peakMax = 0, lastPeak = 0, strideMaxFlex = -999;
float pkEma = 0, ampEma = 0, ivEma = 0, basePk = 0, baseAmp = 0, baseIv = 0;
uint32_t strides = 0, impacts = 0;
bool inPeak = false, calibrating = false, mute = false;
unsigned long peakStartMs = 0, prevStartMs = 0, stepFlashUntil = 0;
unsigned long lastUs = 0;
uint8_t level = 0, lastLevel = 0, impactNew = 0;
uint16_t flags = 0;
unsigned long lastAlertMs = 0, raiseSince = 0, lowSince = 0;
volatile bool calibRequest = false;
bool bleConnected = false;
BLECharacteristic *txChar;

Hold hImp = {0, 0}, hVel = {0, 0}, hLat = {0, 0};
uint8_t holdUpdate(Hold &h, uint8_t nl, unsigned long now, unsigned long ms) {
  if (nl > 0 && nl >= h.lvl) { h.lvl = nl; h.until = now + ms; }
  else if (now >= h.until) h.lvl = nl;
  return h.lvl;
}

// ===================== SORTIES : LED / MOTEUR / BUZZER =====================
void setLed(uint8_t i, bool r, bool g, bool b) {
#if USE_RGB
  bool v[3] = {r, g, b};
  for (int c = 0; c < 3; c++) digitalWrite(LED_PINS[i][c], LED_COMMON_ANODE ? !v[c] : v[c]);
#endif
}
void updateLeds(unsigned long now) {
  bool b2 = (now / 250) % 2, b1 = (now / 500) % 2;
  if (calibrating) return;
  // LED1 : niveau (vert / orange / rouge clignotant)
  if (level == 0) setLed(0, 0, 1, 0); else if (level == 1) setLed(0, 1, 1, 0); else setLed(0, b2, 0, 0);
  // LED2 : BLE (bleu fixe = connecte, clignotant = en attente)
  setLed(1, 0, 0, bleConnected ? 1 : b1);
  // LED3 : erreur capteur (rouge) / foulee detectee (cyan)
  if (!shank.ok) setLed(2, b2, 0, 0);
  else if (now < stepFlashUntil) setLed(2, 0, 1, 1);
  else setLed(2, 0, 0, 0);
}

const uint16_t PAT_WARN[]   = {200};
const uint16_t PAT_DANGER[] = {150, 100, 150, 100, 400};
const uint16_t *pat; uint8_t patLen = 0, patIdx = 0;
bool patActive = false, patBuzzer = false;
unsigned long patT = 0;
void setOut(bool on) {
#if USE_MOTOR
  digitalWrite(PIN_MOTOR, on);
#endif
  digitalWrite(PIN_BUZZER, on && patBuzzer && !mute);
}
void startPattern(const uint16_t *p, uint8_t n, bool buzzer) {
  pat = p; patLen = n; patIdx = 0; patBuzzer = buzzer; patT = millis(); patActive = true; setOut(true);
}
void updatePattern() {
  if (!patActive) return;
  if (millis() - patT >= pat[patIdx]) {
    patIdx++; patT = millis();
    if (patIdx >= patLen) { patActive = false; setOut(false); }
    else setOut(patIdx % 2 == 0);
  }
}

// ===================== CAPTEURS ANALOGIQUES =====================
float readLM35() {
  uint32_t s = 0;
  for (int i = 0; i < 32; i++) s += analogReadMilliVolts(PIN_LM35);
  return (s / 32.0f) / 10.0f;
}
float readVbat() {
  uint32_t s = 0;
  for (int i = 0; i < 16; i++) s += analogReadMilliVolts(PIN_VBAT);
  return (s / 16.0f) * 2.0f / 1000.0f;
}

// ===================== CALIBRATION =====================
// Athlete debout, jambe tendue, immobile ~2 s
void calibrate() {
  calibrating = true;
  setLed(0, 0, 0, 0); setLed(1, 0, 0, 0); setLed(2, 1, 1, 1);
  digitalWrite(PIN_BUZZER, HIGH); delay(100); digitalWrite(PIN_BUZZER, LOW);
  delay(1000);
  Imu *arr[2] = {&shank, &thigh};
  for (int k = 0; k < 2; k++) { arr[k]->bx = arr[k]->by = arr[k]->bz = 0; }
  double s[2][5] = {{0}}; int n[2] = {0, 0};
  for (int i = 0; i < 200; i++) {
    for (int k = 0; k < 2; k++) {
      if (!arr[k]->ok) continue;
      if (mpuRead(*arr[k])) {
        s[k][0] += arr[k]->ax; s[k][1] += arr[k]->az;
        s[k][2] += arr[k]->gx; s[k][3] += arr[k]->gy; s[k][4] += arr[k]->gz; n[k]++;
      }
    }
    delay(5);
  }
  for (int k = 0; k < 2; k++) {
    if (n[k] == 0) continue;
    arr[k]->bx = s[k][2] / n[k]; arr[k]->by = s[k][3] / n[k]; arr[k]->bz = s[k][4] / n[k];
    arr[k]->ref = atan2f(-s[k][0] / n[k], s[k][1] / n[k]) * RAD2DEG;
    arr[k]->raw = arr[k]->ref; arr[k]->ang = 0;
  }
  baseTemp = readLM35(); tempC = baseTemp;
  strides = 0; impacts = 0; inPeak = false; strideMaxFlex = -999;
  pkEma = ampEma = ivEma = basePk = baseAmp = baseIv = 0;
  level = lastLevel = 0; raiseSince = lowSince = 0;
  hImp = {0, 0}; hVel = {0, 0}; hLat = {0, 0};
  calibrating = false;
  setLed(2, 0, 0, 0);
#if USE_MOTOR
  digitalWrite(PIN_MOTOR, HIGH); delay(120); digitalWrite(PIN_MOTOR, LOW);
#endif
  lastUs = micros();
#if USE_AI
  ia_reset_session();   // nouvelle session IA : fusion, reference de la derive de fatigue, fenetres, compteurs
#endif
  Serial.printf("# Calibration OK | tibia ref=%.1f | cuisse ref=%.1f | T0=%.1f C\n", shank.ref, thigh.ref, baseTemp);
}

// ===================== BLE =====================
#define UUID_SVC "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define UUID_RX  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define UUID_TX  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
class SrvCb : public BLEServerCallbacks {
  void onConnect(BLEServer *s) { bleConnected = true; }
  void onDisconnect(BLEServer *s) { bleConnected = false; BLEDevice::startAdvertising(); }
};
class RxCb : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) {
    auto v = c->getValue();
    if (v.length() == 0) return;
    if (v[0] == 'C' || v[0] == 'c') calibRequest = true;
    if (v[0] == 'M' || v[0] == 'm') mute = !mute;
  }
};
void initBLE() {
  BLEDevice::init("WorkSafe");
  BLEDevice::setMTU(185);
  BLEServer *srv = BLEDevice::createServer();
  srv->setCallbacks(new SrvCb());
  BLEService *svc = srv->createService(UUID_SVC);
  txChar = svc->createCharacteristic(UUID_TX, BLECharacteristic::PROPERTY_NOTIFY);
  txChar->addDescriptor(new BLE2902());
  BLECharacteristic *rx = svc->createCharacteristic(UUID_RX, BLECharacteristic::PROPERTY_WRITE);
  rx->setCallbacks(new RxCb());
  svc->start();
  BLEDevice::getAdvertising()->addServiceUUID(UUID_SVC);
  BLEDevice::startAdvertising();
}

// ===================== SETUP =====================
void setup() {
#if USE_AI
  ia_serie_tx_taille();   // tampon d'emission serie plus grand (avant Serial.begin)
#endif
  Serial.begin(115200);
  setCpuFrequencyMhz(80);
#if USE_AI
  {   // IA : interpreteurs, verifications et auto-test AVANT l'attente du capteur tibia (utilisable sans capteur)
    IaOptions o = {};
    o.tache_separee = AI_SEPARATE_TASK; o.decim_mode = AI_DECIM_MODE; o.afficher_serie = AI_PRINT_SERIAL;
    o.afficher_ble = AI_PRINT_BLE; o.autotest_boot = AI_AUTOTEST_BOOT;
#if AI_AUTOTEST_BOOT
    o.autotest = ia_autotest; o.fenetre_test = ia_fenetre_test();
#endif
    ia_setup(o);
  }
#endif
  pinMode(PIN_MOTOR, OUTPUT);  digitalWrite(PIN_MOTOR, LOW);
  pinMode(PIN_BUZZER, OUTPUT); digitalWrite(PIN_BUZZER, LOW);
#if USE_RGB
  for (int i = 0; i < 3; i++) for (int c = 0; c < 3; c++) pinMode(LED_PINS[i][c], OUTPUT);
  for (int i = 0; i < 3; i++) setLed(i, 0, 0, 0);
#endif
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_LM35, ADC_2_5db);
  analogSetPinAttenuation(PIN_VBAT, ADC_11db);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  shank.ok = mpuInit(shank);
  thigh.ok = mpuInit(thigh);
  while (!shank.ok) {
    Serial.println("# MPU tibia (0x68) introuvable : verifier SDA/SCL/AD0->GND");
    for (int i = 0; i < 3; i++) { setLed(2, 1, 0, 0); digitalWrite(PIN_BUZZER, HIGH); delay(100); digitalWrite(PIN_BUZZER, LOW); setLed(2, 0, 0, 0); delay(150); }
    delay(1000);
    shank.ok = mpuInit(shank);
  }
  if (!thigh.ok) Serial.println("# MPU cuisse (0x69) absent : mode 1 capteur (angle = inclinaison du tibia)");
#if USE_AI
  ia_capteur_cuisse(thigh.ok);
#endif
  initBLE();
  Serial.println("# WorkSafe pret : jambe tendue, immobile, calibration en cours...");
  calibrate();
}

// ===================== LOOP =====================
void loop() {
  static unsigned long tSlow = 0, tOut = 0, tLed = 0;
  static uint32_t cnt = 0;
  unsigned long nowUs = micros();
  updatePattern();
#if USE_AI
  ia_commandes();   // T, B, I, P, L sur le port serie (ne bloque jamais)
#endif
  if (nowUs - lastUs < 5000) return;          // boucle a 200 Hz
  float dt = (nowUs - lastUs) / 1e6f; if (dt > 0.05f) dt = 0.005f;
  lastUs = nowUs;
  unsigned long now = millis();

  if (calibRequest) { calibRequest = false; calibrate(); return; }

  if (!mpuRead(shank)) return;
  if (thigh.ok) mpuRead(thigh);
  updateAngle(shank, dt);
  if (thigh.ok) {
    updateAngle(thigh, dt);
    knee = KNEE_SIGN * wrap180(shank.ang - thigh.ang);
    kneeVel = KNEE_SIGN * (shank.gy - thigh.gy);
  } else {
    knee = shank.ang; kneeVel = shank.gy;
  }
  latRate = sqrtf(shank.gx * shank.gx + shank.gz * shank.gz);
  if (knee > strideMaxFlex) strideMaxFlex = knee;

  // ---- Foulees et chocs (accelerometre du tibia) ----
  float mag = sqrtf(shank.ax * shank.ax + shank.ay * shank.ay + shank.az * shank.az);
  if (!inPeak) {
    if (mag > STEP_G && now - peakStartMs > 250) {
      inPeak = true; peakMax = mag; prevStartMs = peakStartMs; peakStartMs = now;
      stepFlashUntil = now + 60;
    }
  } else {
    if (mag > peakMax) peakMax = mag;
    if (mag < STEP_G * 0.8f) {
      inPeak = false; lastPeak = peakMax; strides++;
      float amp = strideMaxFlex; strideMaxFlex = -999;
      unsigned long iv = peakStartMs - prevStartMs;
      if (strides == 1) { pkEma = peakMax; ampEma = amp; }
      else {
        pkEma = 0.9f * pkEma + 0.1f * peakMax;
        ampEma = 0.9f * ampEma + 0.1f * amp;
        if (iv > 200 && iv < 2500) ivEma = (ivEma == 0) ? iv : 0.9f * ivEma + 0.1f * iv;
      }
      if (strides == BASE_STRIDES) { basePk = pkEma; baseAmp = ampEma; baseIv = ivEma; }
      if (peakMax > IMPACT_G) impacts++;
      impactNew = peakMax > PEAK_DANGER_G ? 2 : (peakMax > PEAK_WARN_G ? 1 : 0);
    }
  }

  // ---- Mesures lentes (2 Hz) ----
  if (now - tSlow >= 500) {
    tSlow = now;
    tempC = 0.8f * tempC + 0.2f * readLM35();
#if USE_BATT_SENSE
    vbat = readVbat();
#endif
  }
  float dT = tempC - baseTemp;
  float cadence = ivEma > 0 ? 60000.0f / ivEma : 0;   // foulees/min de la jambe equipee

  // ---- Niveaux par indicateur ----
  uint8_t lv[9] = {0};
  lv[0] = knee > ANGLE_DANGER ? 2 : (knee > ANGLE_WARN ? 1 : 0);
  lv[1] = knee < HYPER_DANGER ? 2 : (knee < HYPER_WARN ? 1 : 0);
  lv[2] = holdUpdate(hVel, fabsf(kneeVel) > VEL_DANGER ? 2 : (fabsf(kneeVel) > VEL_WARN ? 1 : 0), now, 800);
  lv[3] = holdUpdate(hImp, impactNew, now, 3000); impactNew = 0;
  lv[4] = dT >= DT_DANGER ? 2 : (dT >= DT_WARN ? 1 : 0);
  lv[5] = holdUpdate(hLat, latRate > LAT_DANGER ? 2 : (latRate > LAT_WARN ? 1 : 0), now, 800);
  bool running = strides > 0 && (now - peakStartMs < 5000);
  if (running && strides >= BASE_STRIDES && basePk > 0) {
    float pr = (pkEma / basePk - 1.0f) * 100.0f;
    lv[6] = pr >= FAT_PEAK_DANGER ? 2 : (pr >= FAT_PEAK_WARN ? 1 : 0);
    if (baseAmp > 20.0f) {
      float ad = (1.0f - ampEma / baseAmp) * 100.0f;
      lv[7] = ad >= FAT_AMP_DANGER ? 2 : (ad >= FAT_AMP_WARN ? 1 : 0);
    }
    if (baseIv > 0 && ivEma > 0) {
      float cd = (1.0f - baseIv / ivEma) * 100.0f;      // intervalle plus long = cadence plus basse
      lv[8] = cd >= FAT_CAD_DANGER ? 2 : (cd >= FAT_CAD_WARN ? 1 : 0);
    }
  }
  int nD = 0, nW = 0; flags = 0;
  for (int i = 0; i < 9; i++) { if (lv[i] == 2) nD++; if (lv[i] == 1) nW++; if (lv[i]) flags |= (1 << i); }
  uint8_t raw = (nD >= 1 || nW >= 2) ? 2 : (nW == 1 ? 1 : 0);

  // ---- Anti-rebond : montee confirmee 0.5 s, descente confirmee 2 s ----
  if (raw > level) {
    lowSince = 0; if (raiseSince == 0) raiseSince = now;
    if (now - raiseSince >= 500) { level = raw; raiseSince = 0; }
  } else if (raw < level) {
    raiseSince = 0; if (lowSince == 0) lowSince = now;
    if (now - lowSince >= 2000) { level = raw; lowSince = 0; }
  } else { raiseSince = lowSince = 0; }

  // ---- Alertes ----
#if USE_AI
  // POINT D'ANCRAGE FUTUR (IA) : ia_etat_fusion(&type) donne l'etat de l'IA (0 OK, 1 ALERTE(type), 2 non classee).
  // Il n'est PAS utilise : mode ombre, l'IA ne commande ni LED, ni moteur, ni buzzer, ni niveau.
#endif
  if (level > lastLevel) {
    startPattern(level == 2 ? PAT_DANGER : PAT_WARN, level == 2 ? 5 : 1, level == 2 || BUZZER_ON_WARN);
    lastAlertMs = now;
  } else if (level == 2 && now - lastAlertMs > 10000 && !patActive) {
    startPattern(PAT_DANGER, 5, true); lastAlertMs = now;
  }
  lastLevel = level;

  if (now - tLed >= 50) { tLed = now; updateLeds(now); }

  // ---- Log CSV ----
#if USE_AI
  {   // mesure a 50 Hz commune a l'IA et au LOG_CSV (memes valeurs, memes instants)
    const float v[IA_N_CANAUX] = {shank.ax, shank.ay, shank.az, shank.gy, thigh.gy, knee};
    float e50[IA_MAX_ECHANTILLONS][IA_N_CANAUX];
    const int ne = ia_mesure_200hz(nowUs, v, running, calibrating, e50);
#if LOG_CSV
    for (int k = 0; k < ne; k++)
      Serial.printf("%lu,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.2f,%u\n", now, e50[k][0], e50[k][1], e50[k][2],
                    e50[k][3], e50[k][4], e50[k][5], tempC, level);
#else
    (void)ne;
#endif
    (void)cnt;   // compteur de l'ancien LOG_CSV, inutile avec l'IA (la mesure a 50 Hz vient de ia_mesure_200hz)
  }
#else
#if LOG_CSV
  if ((++cnt % LOG_CSV_DIV) == 0)
    Serial.printf("%lu,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.2f,%u\n", now, shank.ax, shank.ay, shank.az,
                  shank.gy, thigh.gy, knee, tempC, level);
#endif
#endif

  // ---- Trame 1 Hz : Serial + BLE ----
  if (now - tOut >= 1000) {
    tOut = now;
    char buf[140];
    int n = snprintf(buf, sizeof(buf), "K:%.1f,V:%.0f,P:%.1f,S:%lu,C:%.0f,T:%.1f,D:%.1f,N:%u,F:%03X",
                     knee, kneeVel, lastPeak, (unsigned long)strides, cadence, tempC, dT, level, flags);
#if USE_BATT_SENSE
    n += snprintf(buf + n, sizeof(buf) - n, ",B:%.2f", vbat);
#endif
#if !LOG_CSV
#if USE_AI && AI_PRINT_SERIAL
    { char ia[48]; ia_champs_trame(ia, sizeof(ia)); Serial.print(buf); Serial.println(ia); }
#else
    Serial.println(buf);
#endif
#endif
#if USE_AI && AI_PRINT_BLE
    // trame BLE + champs IA : tronquee a 180 octets au plus (MTU 185 - 3 octets d'en-tete ATT ; en pratique < 130)
    if (bleConnected) {
      char bb[180]; int m = snprintf(bb, sizeof(bb), "%s", buf);
      m += ia_champs_trame(bb + m, sizeof(bb) - m); if (m > (int)sizeof(bb) - 1) m = sizeof(bb) - 1;
      txChar->setValue((uint8_t *)bb, m); txChar->notify();
    }
    (void)n;
#else
    if (bleConnected) { txChar->setValue((uint8_t *)buf, n); txChar->notify(); }
#endif
  }
}
