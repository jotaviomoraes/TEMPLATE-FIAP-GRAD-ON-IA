#include "DHT.h"

// Definição dos pinos 
#define DHTPIN 15
#define DHTTYPE DHT22
#define PUMP_RELAY_PIN 2
#define LDR_PH_PIN 34
#define BTN_N 12
#define BTN_P 13
#define BTN_K 14

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(PUMP_RELAY_PIN, OUTPUT);
  digitalWrite(PUMP_RELAY_PIN, LOW);

  // Configuração dos botões como INPUT_PULLUP
  pinMode(BTN_N, INPUT_PULLUP);
  pinMode(BTN_P, INPUT_PULLUP);
  pinMode(BTN_K, INPUT_PULLUP);

  Serial.println("FarmTech Solutions - Monitoramento Figo-da-índia");
}

void loop() {
  // 1. Leitura da Umidade (Simulando solo)
  float umidade = dht.readHumidity();

  // 2. Leitura do pH (LDR no pino 34)
  int leituraLdr = analogRead(LDR_PH_PIN);
  float phStatus = (leituraLdr / 4095.0) * 14.0;

  // 3. Leitura dos Nutrientes (LOW = Pressionado)
  bool n_ok = (digitalRead(BTN_N) == LOW);
  bool p_ok = (digitalRead(BTN_P) == LOW);
  bool k_ok = (digitalRead(BTN_K) == LOW);

  // 4. Lógica de Irrigação (Figo-da-índia: Solo seco < 25%)
  if (!isnan(umidade) && umidade < 25.0) {
    digitalWrite(PUMP_RELAY_PIN, HIGH);
  } else {
    digitalWrite(PUMP_RELAY_PIN, LOW);
  }

  // 5. Monitor Serial
  Serial.println("------------------------------------");
  Serial.print("Umidade: "); Serial.print(umidade); Serial.println("%");
  Serial.print("pH da Terra: "); Serial.println(phStatus);
  Serial.print("Nutrientes: N:"); Serial.print(n_ok ? "OK" : "!!");
  Serial.print(" P:"); Serial.print(p_ok ? "OK" : "!!");
  Serial.print(" K:"); Serial.println(k_ok ? "OK" : "!!");
  Serial.print("Bomba: "); Serial.println(digitalRead(PUMP_RELAY_PIN) ? "LIGADA" : "DESLIGADA");

  delay(2000);
}