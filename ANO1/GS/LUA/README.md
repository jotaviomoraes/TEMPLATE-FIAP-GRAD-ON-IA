# 🌑 Lunar He-3 & Solar Wind Monitor v3.0 — Fusion Edition

Monitor de **vento solar** e **mineralogia lunar** com Machine Learning multi-modelo para
estimativa de Hélio-3 (He-3) e detecção de Ilmenita (FeTiO₃).

---

## 📡 Fontes de Dados

| Módulo | Satélite / Missão | API | Auth |
|--------|-------------------|-----|------|
| Vento solar (plasma) | NOAA DSCOVR | `services.swpc.noaa.gov` | Nenhuma |
| Campo magnético (IMF) | NOAA DSCOVR | `services.swpc.noaa.gov` | Nenhuma |
| CME / Flares / Tempestades | ACE, WIND, DSCOVR | `api.nasa.gov/DONKI` | NASA API Key |
| Ilmenita / TiO₂ | Chandrayaan-1 M3 | `cmr.earthdata.nasa.gov` | Earthdata Token |
| Ti, Fe, He-3 | JAXA SELENE/Kaguya | `darts.isas.jaxa.jp` | Nenhuma |
| Datasets gerais | LRO, Lunar Prospector | `pds.mcp.nasa.gov/api` | Nenhuma |

---

## 🔑 Credenciais Necessárias

Antes de instalar, você precisa criar conta em dois serviços **gratuitos** da NASA:

### 1. NASA API Key
> Usada para acessar o DONKI (CMEs, Flares, Tempestades geomagnéticas)

1. Acesse **https://api.nasa.gov**
2. Preencha nome e e-mail → clique em **"Generate API Key"**
3. Você recebe a chave **na hora por e-mail** (leva ~2 minutos)

### 2. NASA Earthdata Token
> Usado para acessar dados do M3 (Chandrayaan-1) e LRO via CMR

1. Acesse **https://urs.earthdata.nasa.gov**
2. Crie uma conta gratuita
3. Após login, vá em **Profile → Generate Token**
4. Copie o token gerado

> ⚠️ O token Earthdata expira após **90 dias**. Quando isso ocorrer, gere um novo e atualize o `.env`.

---

## ⚙️ Instalação

```bash
# 1. Coloque todos os arquivos numa mesma pasta
mkdir lunar_he3_monitor && cd lunar_he3_monitor

# 2. (Recomendado) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Crie o arquivo .env (veja abaixo)
```

### Criando o arquivo `.env`

Crie um arquivo chamado exatamente `.env` (sem nome antes do ponto) **na mesma pasta** que o `lunar_v3.py` e cole o conteúdo abaixo substituindo com suas chaves:

```dotenv
NASA_API_KEY=SUA_CHAVE_NASA_AQUI
EARTHDATA_TOKEN=SEU_TOKEN_EARTHDATA_AQUI
OUTPUT_DIR=./outputs
DATA_WINDOW_DAYS=7
```

> ⚠️ **NUNCA** compartilhe o arquivo `.env` publicamente.
> Se usar Git, adicione `.env` ao seu `.gitignore`.

---

## 🚀 Uso

```bash
# Dashboard completo (padrão)
python lunar_v3.py

# Apenas vento solar
python lunar_v3.py --solar

# Apenas mapa lunar
python lunar_v3.py --lunar

# Apenas benchmark de Machine Learning
python lunar_v3.py --ml

# Exportar todos os dados (CSV + JSON)
python lunar_v3.py --export

# Sem exibir janela gráfica (salva apenas arquivo)
python lunar_v3.py --no-plot

# Combinações
python lunar_v3.py --export --no-plot
```

---

## 📊 O que o programa faz

### Módulo 1 — Vento Solar
- Busca plasma do vento solar em tempo real (velocidade, densidade, temperatura)
- Busca campo magnético interplanetário (IMF/Bz)
- Lista CMEs, flares e tempestades geomagnéticas recentes
- Calcula fluxo de partículas que atinge a Lua (sem escudo magnético)

### Módulo 2 — Machine Learning
Treina e avalia **4 modelos** com validação cruzada KFold (k=5) usando um dataset de calibração que combina valores de referência das missões Apollo 11, 12, 14, 15, 17 e SELENE/Kaguya com amostras sintéticas para demonstração:

| Modelo | Característica |
|--------|---------------|
| Linear Regression | Baseline interpretável |
| Ridge Regression | Linear com regularização L2 |
| Random Forest | Ensemble de árvores, captura não-linearidades |
| Gradient Boosting | Boosting sequencial, maior acurácia preditiva |

Features de entrada: `TiO₂ (%)`, `fluxo solar normalizado`, `latitude absoluta`, `é mare?`  
Target: concentração de **He-3 em ppb**  
Métricas reportadas: R², MAE e RMSE (via cross-validation)

### Módulo 3 — Mineralogia Lunar
- Consulta o CMR por granules do **Moon Mineralogy Mapper (M3)** do Chandrayaan-1 (metadados)
- Consulta dados GRS do **Lunar Prospector** (catálogo)
- Consulta o PDS por datasets de TiO₂ lunar (catálogo)
- Consulta metadados do **SELENE/Kaguya** (JAXA)
- Gera mapa **simulado** de TiO₂/He-3 por região (gaussianas calibradas com literatura) e aplica o melhor modelo de ML do Módulo 2

> O mapa não processa os arquivos brutos das APIs — usa regiões catalogadas e estimativas demonstrativas.

### Módulo 4 — Dashboard Visual
Painel com 6 painéis integrados:
- Gráfico de velocidade do vento solar (7 dias)
- Gráfico do campo Bz (norte = azul, sul = laranja)
- Painel de status com alertas de CME/flares
- Gráfico comparativo dos modelos de ML
- Mapa lunar colorido por concentração de He-3
- Tabela ranqueada das regiões com maior potencial de prospecção

### Módulo 5 — Exportação
Salva em `./outputs/`:
- `solar_wind_plasma.csv`
- `solar_wind_mag.csv`
- `cme_events.json`
- `flare_events.json`
- `lunar_ilmenite_regions.csv`
- `ml_metrics.json`
- `apollo_calibration_data.csv`
- `dashboard_v3_YYYYMMDD_HHMMSS.png`

---

## 🗺️ Regiões de Alta Ilmenita / He-3

| Região | TiO₂ (%) | He-3 est. (ppb) |
|--------|----------|-----------------|
| Mare Tranquillitatis | 11.8 | ~28 |
| Mare Serenitatis | 7.2 | ~17 |
| Mare Imbrium | 5.6 | ~13 |
| Mare Crisium | 5.1 | ~12 |
| Mare Humorum | 4.0 | ~10 |

---

## 📚 Referências

- Haskin & Warren (1991): correlação TiO₂ × implantação He-3
- Fa & Jin (2007): modelo de implantação de vento solar na Lua
- Pieters et al. (2009): M3 — Chandrayaan-1
- Ogawa et al. (2011): SELENE/Kaguya GRS composição elemental
- NASA DONKI: https://kauai.ccmc.gsfc.nasa.gov/DONKI/
- NOAA Space Weather: https://www.swpc.noaa.gov/
- NASA CMR: https://cmr.earthdata.nasa.gov/
- JAXA DARTS: https://darts.isas.jaxa.jp/planet/pdap/selene/
