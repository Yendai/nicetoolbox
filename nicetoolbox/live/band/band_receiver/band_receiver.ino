/*
  Wristband cue receiver: XIAO nRF52840, Bluetooth LE in, motor buzz out.

  The web page (python -m nicetoolbox.live --ui) connects to this board over Bluetooth LE and
  sends one-letter commands on the Nordic UART service (NUS). The band buzzes only when a cue
  STARTS (change-only protocol): crossing the arms buzzes once, uncrossing is silent.

  Commands (also accepted from the Arduino Serial Monitor, 115200 baud, for bench tests):
    A  arms crossed: cue started    -> cue pattern on the dot motor (two short buzzes)
    a  arms uncrossed: cue ended    -> no buzz, state only
    T  test buzz                    -> one long buzz on the dot motor
    S  test the status motor        -> one long buzz on the status motor
    H  heartbeat from the page      -> no buzz; after the first one, silence for
                                       FAULT_TIMEOUT_MS counts as a fault
    X  page is disconnecting on purpose: no fault buzz for this disconnect
    ?  print state
  The band answers each command over Bluetooth with a line such as "ok A".

  Faults (the system stopped without saying so: laptop asleep, page closed, Python crashed):
  three short buzzes on the status motor and a red LED, once per fault.

  LED: blue blinking = waiting for the page, green = connected, red = fault.

  Circuit: same as the troubleshoot sketch. Dot motor on D0 through the NPN driver (1k base,
  10k base to GND, diode across the motor). Status motor on D9 with the same driver; if it is
  not wired yet, the status buzzes simply do nothing.

  Arduino IDE: board "Seeed nRF52 Boards > Seeed XIAO nRF52840" (not mbed-enabled). The
  Bluefruit library ships with that board package, nothing extra to install.
*/

#include <Adafruit_TinyUSB.h>   // this board package needs it for Serial over USB
#include <bluefruit.h>

const char *BAND_NAME = "HapticBand";      // the page looks for this name
const int DOT_PIN = D0;                    // dot 1 (pin map: D0 D1 D3 D4 D5 D8 = dots 1-6)
const int STATUS_PIN = D9;                 // status motor
const unsigned long FAULT_TIMEOUT_MS = 30000;  // page sends H every 10 s

#ifdef OUTPUT_H0H1
const uint32_t MOTOR_OUT_MODE = OUTPUT_H0H1;   // high drive, as in the troubleshoot sketch
#else
const uint32_t MOTOR_OUT_MODE = OUTPUT;
#endif

// ---- Patterns: on/off times in ms, ending with 0. Tune these with Panashe. ----
const uint16_t PATTERN_ARMS[] = {200, 150, 200, 0};
const uint16_t PATTERN_TEST[] = {600, 0};
const uint16_t PATTERN_FAULT[] = {150, 150, 150, 150, 150, 0};
const uint16_t PATTERN_CONNECTED[] = {120, 0};

// A pattern player that never blocks: loop() calls update() and it switches the motor
// at the right moments. Blocking with delay() would stall Bluetooth handling.
struct Motor {
  int pin;
  const uint16_t *pattern;
  uint8_t step;
  unsigned long stepStarted;

  Motor(int p) : pin(p), pattern(nullptr), step(0), stepStarted(0) {}

  void begin() {
    pinMode(pin, MOTOR_OUT_MODE);
    digitalWrite(pin, LOW);
  }
  void play(const uint16_t *p) {   // a new pattern replaces one that is still playing
    pattern = p;
    step = 0;
    stepStarted = millis();
    digitalWrite(pin, HIGH);       // even steps are "on", odd steps are "off"
  }
  void update() {
    if (!pattern) return;
    if (millis() - stepStarted < pattern[step]) return;
    step++;
    stepStarted = millis();
    if (pattern[step] == 0) {      // end of the pattern
      pattern = nullptr;
      digitalWrite(pin, LOW);
      return;
    }
    digitalWrite(pin, step % 2 == 0 ? HIGH : LOW);
  }
};

Motor dot(DOT_PIN);
Motor statusMotor(STATUS_PIN);
BLEUart bleuart;

bool armsCrossed = false;
bool heartbeatSeen = false;        // fault timeout only runs once the page sends heartbeats
bool faulted = false;
bool goodbye = false;              // page said X before disconnecting
unsigned long lastMessage = 0;
volatile bool justConnected = false;
volatile bool justDisconnected = false;

void setRgb(bool r, bool g, bool b) {   // LOW turns the onboard LED colours on
  digitalWrite(LED_RED, r ? LOW : HIGH);
  digitalWrite(LED_GREEN, g ? LOW : HIGH);
  digitalWrite(LED_BLUE, b ? LOW : HIGH);
}

void reply(const char *text) {
  Serial.println(text);
  if (Bluefruit.connected()) bleuart.println(text);
}

void fault(const char *why) {
  if (faulted) return;
  faulted = true;
  statusMotor.play(PATTERN_FAULT);
  Serial.print("FAULT: ");
  Serial.println(why);
}

void handle(char c) {
  lastMessage = millis();
  if (c != '?' && c != '\n' && c != '\r' && c != ' ') faulted = false;   // any message clears a fault
  switch (c) {
    case 'A':
      if (!armsCrossed) dot.play(PATTERN_ARMS);   // change-only: buzz when the cue starts
      armsCrossed = true;
      reply("ok A");
      break;
    case 'a': armsCrossed = false; reply("ok a"); break;
    case 'T': dot.play(PATTERN_TEST); reply("ok T"); break;
    case 'S': statusMotor.play(PATTERN_TEST); reply("ok S"); break;
    case 'H': heartbeatSeen = true; reply("ok H"); break;
    case 'X': goodbye = true; heartbeatSeen = false; reply("ok X"); break;
    case '?':
      Serial.print("connected="); Serial.print(Bluefruit.connected());
      Serial.print(" armsCrossed="); Serial.print(armsCrossed);
      Serial.print(" heartbeat="); Serial.print(heartbeatSeen);
      Serial.print(" faulted="); Serial.println(faulted);
      if (Bluefruit.connected()) bleuart.println(armsCrossed ? "state A" : "state a");
      break;
    case '\n': case '\r': case ' ': break;
    default:
      Serial.print("Unknown command '"); Serial.print(c); Serial.println("'");
      if (Bluefruit.connected()) bleuart.println("unknown");
  }
}

// Bluetooth callbacks run outside loop(), so they only set flags; loop() does the work.
void onConnect(uint16_t conn_handle) { justConnected = true; }
void onDisconnect(uint16_t conn_handle, uint8_t reason) { justDisconnected = true; }

void startAdvertising() {
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);   // the page filters on this service
  Bluefruit.ScanResponse.addName();            // and on the name
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);  // fast at first, then slower to save battery
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);              // advertise until someone connects
}

void setup() {
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  setRgb(0, 0, 0);
  dot.begin();
  statusMotor.begin();

  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);   // do not wait forever on battery

  Bluefruit.begin();
  Bluefruit.setTxPower(4);
  Bluefruit.setName(BAND_NAME);
  Bluefruit.Periph.setConnectCallback(onConnect);
  Bluefruit.Periph.setDisconnectCallback(onDisconnect);
  bleuart.begin();
  startAdvertising();

  Serial.println();
  Serial.println("=== Wristband cue receiver ===");
  Serial.print("Advertising as "); Serial.println(BAND_NAME);
  Serial.println("Commands: A arms cue | a arms end | T test buzz | S status buzz | ? state");
}

void loop() {
  if (justConnected) {
    justConnected = false;
    goodbye = false;
    faulted = false;
    lastMessage = millis();
    statusMotor.play(PATTERN_CONNECTED);
    Serial.println("Page connected");
  }
  if (justDisconnected) {
    justDisconnected = false;
    Serial.println("Page disconnected");
    if (!goodbye) fault("lost the page");
    heartbeatSeen = false;
    armsCrossed = false;
  }

  while (bleuart.available()) handle((char)bleuart.read());
  while (Serial.available()) handle((char)Serial.read());

  if (Bluefruit.connected() && heartbeatSeen && millis() - lastMessage > FAULT_TIMEOUT_MS) {
    fault("no message from the page");
  }

  dot.update();
  statusMotor.update();

  if (faulted) setRgb(1, 0, 0);
  else if (Bluefruit.connected()) setRgb(0, 1, 0);
  else setRgb(0, 0, (millis() / 500) % 2);   // blink blue while waiting
}
