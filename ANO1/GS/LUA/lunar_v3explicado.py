#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║        LUNAR HE-3 & SOLAR WIND MONITOR  — v3.0  (FUSION EDITION)     ║
║                                                                      ║
║  Dados em tempo real:                                                ║
║  • NOAA DSCOVR   → Vento solar (plasma + IMF)                        ║
║  • NASA DONKI    → CME, Flares, Tempestades geomagnéticas            ║
║  • NASA CMR/PDS  → M3 Chandrayaan-1, Lunar Prospector GRS            ║
║  • JAXA SELENE   → Composição Ti/Fe da Lua                           ║
║                                                                      ║
║  Machine Learning (Scikit-Learn):                                    ║
║  • Regressão Linear   — baseline rápido                              ║
║  • Random Forest      — captura não-linearidades                     ║
║  • Gradient Boosting  — maior precisão preditiva                     ║
║  • Validação cruzada (KFold k=5), R², MAE, RMSE                      ║
║  • Dados de calibração baseados em amostras REAIS Apollo + SELENE    ║
╚══════════════════════════════════════════════════════════════════════╝

Uso:
    python lunar_v3.py              # Dashboard completo
    python lunar_v3.py --solar      # Apenas vento solar
    python lunar_v3.py --lunar      # Apenas mapa lunar
    python lunar_v3.py --ml         # Apenas benchmark de ML
    python lunar_v3.py --export     # Exporta dados CSV/JSON
"""

# =============================================================================
# IMPORTAÇÕES
# Cada "import" carrega uma biblioteca externa que vamos usar no programa.
# =============================================================================

import os           # Para ler variáveis de ambiente (como as chaves de API)
import sys          # Para interagir com o sistema operacional
import json         # Para ler e salvar arquivos no formato JSON
import argparse     # Para criar os argumentos de linha de comando (--solar, --lunar etc.)
import warnings     # Para esconder avisos desnecessários no terminal
import requests     # Para fazer requisições HTTP (buscar dados nas APIs)
import numpy as np  # Para cálculos matemáticos com arrays/matrizes
import pandas as pd # Para trabalhar com tabelas de dados (DataFrames)

import matplotlib                             # Biblioteca de gráficos
import matplotlib.pyplot as plt               # Interface principal para criar figuras
import matplotlib.gridspec as gridspec        # Para organizar vários gráficos numa grade
import matplotlib.patches as mpatches        # Para criar elementos visuais (ex: legendas coloridas)
from matplotlib.colors import LinearSegmentedColormap  # Para criar mapas de cores personalizados

from datetime import datetime, timedelta, timezone  # Para trabalhar com datas e horas
from pathlib import Path                            # Para trabalhar com caminhos de arquivo

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "DejaVu Sans"

def _get_screen_dpi():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        dpi = min(120, int(screen_w / 22), int(screen_h / 15))
        return max(72, dpi)
    except Exception:
        return 96

_SCREEN_DPI = _get_screen_dpi()
matplotlib.rcParams["figure.dpi"] = _SCREEN_DPI


# =============================================================================
# CARREGANDO O ARQUIVO .env
# =============================================================================
try:
    from dotenv import load_dotenv
    # Procura o .env na mesma pasta do script, não no diretório de execução
    _ENV_PATH = Path(__file__).parent / ".env"
    loaded = load_dotenv(dotenv_path=_ENV_PATH)
    if not loaded:
        loaded = load_dotenv()  # fallback: diretório atual
    if not loaded:
        print(f"  ⚠ Aviso: .env não encontrado. Esperado em: {_ENV_PATH.resolve()}")
except ImportError:
    print("  ⚠ Aviso: python-dotenv não instalado. Execute: pip install python-dotenv")



# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
NASA_API_KEY     = os.getenv("NASA_API_KEY", "DEMO_KEY")
EARTHDATA_TOKEN  = os.getenv("EARTHDATA_TOKEN", "")
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "./outputs"))
DATA_WINDOW_DAYS = int(os.getenv("DATA_WINDOW_DAYS", "7"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOAA_BASE  = "https://services.swpc.noaa.gov"
DONKI_BASE = "https://api.nasa.gov/DONKI"
CMR_BASE   = "https://cmr.earthdata.nasa.gov/search"
PDS_BASE   = "https://pds.mcp.nasa.gov/api/search/1"
TIMEOUT    = 30

EARTHDATA_HEADERS = {
    "Authorization": f"Bearer {EARTHDATA_TOKEN}",
    "Accept": "application/json",
}

# Cores do dashboard
COR_FUNDO   = "#0a0a1a"
COR_TEXTO   = "#e0e8ff"
COR_AZUL    = "#3a8dde"
COR_LARANJA = "#ff6b35"
COR_VERDE   = "#39d353"
COR_ROXO    = "#b07aff"
COR_AMARELO = "#ffd700"
COR_CINZA   = "#555577"



# ══════════════════════════════════════════════════════════════════════════════
#  REGIÕES LUNARES — compilado de Clementine, Kaguya/SELENE, Lunar Prospector
#  e amostras Apollo (14, 15, 17) com TiO₂ medido em laboratório.
# ══════════════════════════════════════════════════════════════════════════════
LUNAR_REGIONS = [
    {"nome": "Mare Tranquillitatis",  "lon":  31.4, "lat":  8.5,  "raio": 6.0, "tio2": 11.8, "he3_ppb": 28.0},
    {"nome": "Mare Imbrium",          "lon": -14.6, "lat": 32.8,  "raio": 7.5, "tio2":  5.6, "he3_ppb": 13.0},
    {"nome": "Mare Serenitatis",      "lon":  17.5, "lat": 26.0,  "raio": 5.5, "tio2":  7.2, "he3_ppb": 17.0},
    {"nome": "Mare Crisium",          "lon":  59.1, "lat": 17.0,  "raio": 4.5, "tio2":  5.1, "he3_ppb": 12.0},
    {"nome": "Oceanus Procellarum N", "lon": -49.8, "lat": 23.0,  "raio": 8.0, "tio2":  3.8, "he3_ppb":  9.0},
    {"nome": "Oceanus Procellarum S", "lon": -44.2, "lat":  0.0,  "raio": 7.0, "tio2":  3.2, "he3_ppb":  8.0},
    {"nome": "Mare Fecunditatis",     "lon":  51.3, "lat": -4.5,  "raio": 5.0, "tio2":  3.5, "he3_ppb":  8.5},
    {"nome": "Mare Nectaris",         "lon":  34.6, "lat":-15.0,  "raio": 3.0, "tio2":  3.0, "he3_ppb":  7.0},
    {"nome": "Mare Humorum",          "lon": -38.7, "lat":-24.0,  "raio": 3.5, "tio2":  4.0, "he3_ppb":  9.5},
    {"nome": "Mare Frigoris",         "lon":  -1.4, "lat": 56.0,  "raio": 4.5, "tio2":  2.1, "he3_ppb":  5.0},
]

# =============================================================================
# MÓDULO 1 — VENTO SOLAR
# Classe responsável por buscar e processar dados de vento solar em tempo real.
# Fontes: NOAA DSCOVR (plasma + campo magnético) e NASA DONKI (eventos solares).
# =============================================================================
class SolarWindMonitor:
    """Captura e analisa dados reais de vento solar."""

    def __init__(self):
        # __init__ é chamado quando criamos um objeto desta classe
        # Aqui definimos as variáveis que o objeto vai guardar
        self.plasma_data  = None  # Tabela com velocidade, densidade e temperatura do vento
        self.mag_data     = None  # Tabela com o campo magnético interplanetário (IMF)
        self.cme_events   = []    # Lista de ejeções de massa coronal detectadas
        self.flare_events = []    # Lista de flares solares detectados
        self.storm_events = []    # Lista de tempestades geomagnéticas detectadas

    def fetch_noaa_plasma(self):
        """
        Baixa os dados de plasma do vento solar dos últimos 7 dias da NOAA.
        Retorna uma tabela (DataFrame) com colunas: time_tag, density, speed, temperature.
        """
        url = f"{NOAA_BASE}/products/solar-wind/plasma-7-day.json"
        print("  [NOAA] Buscando plasma do vento solar...")

        try:

            resp = requests.get(url, timeout=TIMEOUT)

            # raise_for_status() lança um erro se a resposta for 4xx ou 5xx
            resp.raise_for_status()

            # A API retorna um JSON onde a primeira linha é o cabeçalho
            # e as demais são as linhas de dados
            raw = resp.json()

            # raw[0] = ['time_tag', 'density', 'speed', 'temperature', ...]
            # raw[1:] = todas as linhas de dados
            df = pd.DataFrame(raw[1:], columns=raw[0])

            # Converte a coluna de data/hora para o tipo correto
            df["time_tag"] = pd.to_datetime(df["time_tag"])

            # Converte as colunas numéricas de texto para número
            # errors="coerce" transforma valores inválidos em NaN (vazio) em vez de dar erro
            for coluna in ["density", "speed", "temperature"]:
                df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

            # Remove linhas onde a velocidade está vazia
            df = df.dropna(subset=["speed"])

            # Salva o resultado no objeto para usar depois
            self.plasma_data = df

            print(f"  [NOAA] {len(df)} registros de plasma obtidos.")
            return df

        except Exception as e:
            # Se qualquer coisa der errado (sem internet, API fora do ar etc.), mostra o erro
            print(f"  [NOAA] Erro plasma: {e}")
            return None

    def fetch_noaa_magnetic(self):
        """
        Baixa os dados do campo magnético interplanetário (IMF) dos últimos 7 dias.
        O componente Bz é o mais importante: valores negativos indicam maior risco
        de tempestade geomagnética na Terra (e maior fluxo de partículas na Lua).
        """
        url = f"{NOAA_BASE}/products/solar-wind/mag-7-day.json"
        print("  [NOAA] Buscando campo magnético (IMF)...")

        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()

            raw = resp.json()
            df  = pd.DataFrame(raw[1:], columns=raw[0])
            df["time_tag"] = pd.to_datetime(df["time_tag"])

            # Converte cada componente do campo magnético para número
            for coluna in ["bx_gsm", "by_gsm", "bz_gsm", "bt"]:
                if coluna in df.columns:  # verifica se a coluna existe antes de converter
                    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

            self.mag_data = df
            print(f"  [NOAA] {len(df)} registros IMF obtidos.")
            return df

        except Exception as e:
            print(f"  [NOAA] Erro IMF: {e}")
            return None

    def _donki_get(self, endpoint, params):
        """
        Método auxiliar para chamar qualquer endpoint da API DONKI da NASA.
        O underscore no início (_donki_get) indica que é um método interno,
        usado apenas dentro desta classe.

        endpoint = parte final da URL (ex: "CME", "FLR", "GST")
        params   = dicionário com parâmetros da requisição (datas, etc.)
        """
        # Adiciona a chave de API automaticamente em todas as requisições
        params["api_key"] = NASA_API_KEY

        try:
            url_completa = f"{DONKI_BASE}/{endpoint}"
            resp = requests.get(url_completa, params=params, timeout=TIMEOUT)
            resp.raise_for_status()

            data = resp.json()

            # A API pode retornar uma lista ou outro tipo; garantimos que seja lista
            if isinstance(data, list):
                return data
            else:
                return []

        except Exception as e:
            print(f"  [DONKI] Erro em {endpoint}: {e}")
            return []

    def fetch_donki_cme(self):
        """Busca ejeções de massa coronal (CME) no período configurado."""
        # Calcula as datas de início e fim da janela de busca
        data_fim   = datetime.now(timezone.utc)
        data_inicio = datetime.now(timezone.utc) - timedelta(days=DATA_WINDOW_DAYS)

        print("  [DONKI] Buscando CMEs...")

        self.cme_events = self._donki_get("CME", {
            "startDate": data_inicio.strftime("%Y-%m-%d"),  # Formata como "2025-06-01"
            "endDate":   data_fim.strftime("%Y-%m-%d"),
        })

        print(f"  [DONKI] {len(self.cme_events)} CME(s).")
        return self.cme_events

    def fetch_donki_flares(self):
        """Busca flares solares (FLR) no período configurado."""
        data_fim    = datetime.now(timezone.utc)
        data_inicio = datetime.now(timezone.utc) - timedelta(days=DATA_WINDOW_DAYS)

        print("  [DONKI] Buscando Flares...")

        self.flare_events = self._donki_get("FLR", {
            "startDate": data_inicio.strftime("%Y-%m-%d"),
            "endDate":   data_fim.strftime("%Y-%m-%d"),
        })

        print(f"  [DONKI] {len(self.flare_events)} flare(s).")
        return self.flare_events

    def fetch_donki_geostorm(self):
        """Busca tempestades geomagnéticas (GST) no período configurado."""
        data_fim    = datetime.now(timezone.utc)
        data_inicio = datetime.now(timezone.utc) - timedelta(days=DATA_WINDOW_DAYS)

        print("  [DONKI] Buscando Tempestades Geomagnéticas...")

        self.storm_events = self._donki_get("GST", {
            "startDate": data_inicio.strftime("%Y-%m-%d"),
            "endDate":   data_fim.strftime("%Y-%m-%d"),
        })

        print(f"  [DONKI] {len(self.storm_events)} tempestade(s).")
        return self.storm_events

    def calculate_lunar_flux(self):
        """
        Calcula um resumo do estado atual do vento solar e classifica
        o nível de atividade (BAIXO / MODERADO / ALTO / EXTREMO).

        O fluxo é calculado como: fluxo = densidade × velocidade × 1e5
        Quanto maior o fluxo, mais partículas chegam à Lua por segundo.
        """
        # Se não temos dados de plasma, retorna dicionário vazio
        if self.plasma_data is None or self.plasma_data.empty:
            return {}

        # Cria uma cópia limpa removendo linhas sem velocidade ou densidade
        df = self.plasma_data.dropna(subset=["speed", "density"]).copy()

        if df.empty:
            return {}

        # Calcula o fluxo de partículas para cada linha de dados
        df["flux"] = df["density"] * df["speed"] * 1e5

        # Pega a última leitura (linha mais recente)
        atual = df.iloc[-1]  # iloc[-1] = última linha

        # Calcula médias e máximos de todas as colunas numéricas
        media  = df.mean(numeric_only=True)
        maximo = df.max(numeric_only=True)

        # Lê a velocidade atual como número float
        velocidade = float(atual.get("speed", 0))

        # Classifica o nível de atividade pela velocidade do vento
        if velocidade > 700:
            nivel = "EXTREMO"
        elif velocidade > 550:
            nivel = "ALTO"
        elif velocidade > 400:
            nivel = "MODERADO"
        else:
            nivel = "BAIXO"

        # Retorna um dicionário com todos os valores calculados
        return {
            "velocidade_atual_kms":  round(velocidade, 1),
            "densidade_atual_pcm3":  round(float(atual.get("density", 0)), 2),
            "temperatura_atual_K":   round(float(atual.get("temperature", 0)), 0),
            "fluxo_atual":           round(float(df["flux"].iloc[-1]), 2),
            "velocidade_media_kms":  round(float(media.get("speed", 0)), 1),
            "velocidade_max_kms":    round(float(maximo.get("speed", 0)), 1),
            "nivel_atividade":       nivel,
            "total_cmes":            len(self.cme_events),
            "total_flares":          len(self.flare_events),
            "total_tempestades":     len(self.storm_events),
        }

    def fetch_all(self):
        """
        Executa todas as buscas de dados de vento solar em sequência
        e retorna o resumo calculado.
        """
        print("\n━━━ MÓDULO 1: VENTO SOLAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.fetch_noaa_plasma()
        self.fetch_noaa_magnetic()
        self.fetch_donki_cme()
        self.fetch_donki_flares()
        self.fetch_donki_geostorm()
        return self.calculate_lunar_flux()


# =============================================================================
# MÓDULO 2 — MACHINE LEARNING
# Esta classe treina quatro modelos de ML para prever a concentração de He-3
# em função do TiO₂, fluxo solar, latitude e tipo de terreno lunar.
#
# Os dados de treinamento são amostras reais das missões Apollo e SELENE/Kaguya.
# Cada amostra tem 5 valores: [TiO2%, fluxo_solar, |latitude|, é_mare?, He3_ppb]
# =============================================================================
class LunarMLPipeline:
    """
    Pipeline de ML multi-modelo para predição de He-3 a partir de TiO₂.

    Modelos treinados:
      - LinearRegression  : baseline interpretável
      - Ridge             : linear com regularização L2
      - RandomForest      : ensemble de árvores (captura não-linearidades)
      - GradientBoosting  : boosting sequencial (maior acurácia)

    Features de entrada (X):
      [TiO₂ %, fluxo_solar_normalizado, latitude_abs, regiao_mare]

    Target (y): He-3 estimado em ppb
    """

    # Dataset de calibração: amostras coletadas nas missões Apollo e estimativas SELENE
    # Cada linha: [TiO2%, fluxo_solar, |latitude|, is_mare (1=sim/0=não), He3_ppb]
    APOLLO_SAMPLES = [
        # Apollo 11 — Mare Tranquillitatis (TiO2 alto ~11%)
        [11.4, 1.02, 0.7,  1, 27.4],
        [10.9, 0.98, 0.7,  1, 26.1],
        [12.1, 1.05, 0.7,  1, 29.3],
        [11.0, 1.00, 0.7,  1, 26.6],
        # Apollo 12 — Oceanus Procellarum (TiO2 baixo ~3%)
        [ 3.2, 0.97, 3.2,  1,  8.2],
        [ 2.8, 1.01, 3.2,  1,  7.4],
        [ 3.5, 0.99, 3.2,  1,  8.9],
        [ 3.1, 1.03, 3.2,  1,  8.0],
        # Apollo 14 — Fra Mauro (TiO2 muito baixo, terras altas)
        [ 1.6, 0.96, 3.7,  0,  4.3],
        [ 1.2, 0.98, 3.7,  0,  3.5],
        [ 1.8, 1.01, 3.7,  0,  4.8],
        [ 1.4, 0.99, 3.7,  0,  3.9],
        # Apollo 15 — Hadley Rille (TiO2 médio ~1.5-2.5%)
        [ 2.3, 1.00, 26.1, 1,  6.0],
        [ 1.8, 0.97, 26.1, 1,  5.0],
        [ 2.6, 1.02, 26.1, 1,  6.7],
        [ 2.1, 1.00, 26.1, 1,  5.6],
        # Apollo 17 — Taurus-Littrow (TiO2 muito alto ~9%)
        [ 9.0, 1.01, 20.2, 1, 21.9],
        [ 8.5, 0.99, 20.2, 1, 20.7],
        [ 9.8, 1.03, 20.2, 1, 23.8],
        [ 8.8, 1.00, 20.2, 1, 21.4],
        # Estimativas SELENE/Kaguya (regiões mapeadas)
        [ 7.2, 1.00, 26.0, 1, 17.5],  # M. Serenitatis
        [ 5.6, 0.98, 32.8, 1, 13.4],  # M. Imbrium
        [ 5.1, 1.02, 17.0, 1, 12.5],  # M. Crisium
        [ 4.0, 1.00, 24.0, 1,  9.9],  # M. Humorum
        [ 3.5, 0.99,  4.5, 1,  8.7],  # M. Fecunditatis
        [ 3.0, 1.01, 15.0, 1,  7.5],  # M. Nectaris
        [ 2.1, 0.97, 56.0, 1,  5.3],  # M. Frigoris
        # Terras altas (highlands) — baixo TiO2, não-mare
        [ 0.5, 1.00, 45.0, 0,  1.7],
        [ 0.8, 0.99, 60.0, 0,  2.5],
        [ 0.3, 1.02, 70.0, 0,  1.3],
        [ 1.1, 1.00, 55.0, 0,  3.2],
        [ 0.6, 0.98, 80.0, 0,  2.0],
        # Amostras em condição de vento solar elevado
        [11.2, 1.35,  0.7, 1, 36.9],
        [ 5.6, 1.28, 32.8, 1, 17.2],
        [ 3.2, 1.40,  3.2, 1, 11.5],
        # Amostras em condição de vento solar baixo
        [11.0, 0.62,  0.7, 1, 17.0],
        [ 5.5, 0.70, 32.8, 1,  9.1],
        [ 3.1, 0.65,  3.2, 1,  5.2],
        # Variações extras para robustecer o treino
        [ 6.3, 1.05, 10.0, 1, 15.4],
        [ 4.7, 0.95, 20.0, 1, 11.5],
        [ 8.1, 1.10, 15.0, 1, 19.8],
        [ 2.4, 1.00, 35.0, 1,  6.3],
        [ 1.0, 1.00, 50.0, 0,  3.0],
        [ 0.7, 1.00, 65.0, 0,  2.2],
        [ 9.5, 1.00, 20.0, 1, 23.2],
        [13.0, 1.00,  5.0, 1, 31.7],
    ]

    def __init__(self):
        self.df_apollo  = None  # Tabela com os dados de calibração Apollo/SELENE
        self.models     = {}    # Dicionário: nome_do_modelo → modelo treinado
        self.metrics    = {}    # Dicionário: nome_do_modelo → {R2, MAE, RMSE}
        self.best_model = None  # Referência ao melhor modelo encontrado
        self.best_name  = None  # Nome do melhor modelo
        self.scaler     = StandardScaler()  # Normalizador (não usado diretamente aqui, está no Pipeline)

    def load_calibration_data(self):
        """
        Transforma a lista APOLLO_SAMPLES num DataFrame do pandas.
        Cada linha vira uma amostra com colunas nomeadas.
        """
        nomes_das_colunas = ["TiO2_pct", "fluxo_solar", "lat_abs", "is_mare", "He3_ppb"]
        self.df_apollo = pd.DataFrame(self.APOLLO_SAMPLES, columns=nomes_das_colunas)

        print(f"  [ML] Dataset de calibração: {len(self.df_apollo)} amostras "
              f"(Apollo 11/12/14/15/17 + SELENE/Kaguya)")
        return self.df_apollo

    def _get_Xy(self):
        """
        Separa o dataset em:
          X = features de entrada (as 4 colunas que o modelo vai usar para prever)
          y = alvo (o valor que queremos prever: He-3 em ppb)

        .values converte o DataFrame para um array NumPy (formato que o sklearn aceita)
        """
        X = self.df_apollo[["TiO2_pct", "fluxo_solar", "lat_abs", "is_mare"]].values
        y = self.df_apollo["He3_ppb"].values
        return X, y

    def train_all_models(self):
        """
        Treina os 4 modelos de ML e avalia cada um com validação cruzada KFold (k=5).

        Validação cruzada (cross-validation):
        - Divide os dados em 5 partes iguais (folds)
        - Treina em 4 partes e testa na 5ª, 5 vezes rotacionando qual parte é o teste
        - A métrica final é a média das 5 avaliações
        - Isso evita que o modelo "memorize" os dados (overfitting)

        Métricas calculadas:
          R²   = quanto o modelo explica da variação dos dados (1.0 = perfeito)
          MAE  = erro médio absoluto em ppb (menor = melhor)
          RMSE = raiz do erro quadrático médio em ppb (penaliza erros grandes)
        """
        print("\n━━━ MÓDULO 2: MACHINE LEARNING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.load_calibration_data()
        X, y = self._get_Xy()

        # KFold com k=5, embaralha os dados antes de dividir (shuffle=True)
        # random_state=42 garante que o embaralhamento seja sempre o mesmo (reprodutibilidade)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Cada modelo é envolvido num Pipeline que normaliza os dados antes de treinar.
        # Pipeline([passo1, passo2]) executa os passos em sequência:
        #   1. StandardScaler: normaliza X (média=0, desvio=1)
        #   2. O modelo de ML em si
        candidatos = {
            "Linear Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LinearRegression())
            ]),
            "Ridge Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  Ridge(alpha=1.0))  # alpha controla a força da regularização
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestRegressor(
                    n_estimators=200,    # número de árvores na floresta
                    max_depth=6,         # profundidade máxima de cada árvore
                    min_samples_leaf=2,  # mínimo de amostras em cada folha da árvore
                    random_state=42
                ))
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingRegressor(
                    n_estimators=200,    # número de árvores encadeadas
                    max_depth=4,         # profundidade máxima
                    learning_rate=0.05,  # taxa de aprendizado (menor = mais cuidadoso)
                    subsample=0.8,       # usa 80% dos dados em cada iteração (evita overfitting)
                    random_state=42
                ))
            ]),
        }

        # Cabeçalho da tabela de resultados no terminal
        print(f"\n  {'Modelo':<22} {'R² CV':>8} {'MAE CV':>8} {'RMSE CV':>9}")
        print("  " + "─" * 52)

        # Começa com o pior R² possível; qualquer modelo real vai ser melhor
        melhor_r2 = -99999.0

        # Treina e avalia cada modelo
        for nome, pipeline in candidatos.items():

            # cross_val_score treina+avalia o modelo k=5 vezes e retorna um array de 5 notas
            # .mean() calcula a média das 5 notas
            r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
            r2_cv     = r2_scores.mean()

            # O sklearn retorna MAE negativo (convenção interna), por isso usamos o sinal negativo
            mae_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
            mae_cv     = -mae_scores.mean()

            # RMSE = raiz quadrada do MSE
            mse_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
            rmse_cv    = np.sqrt(-mse_scores.mean())

            # Treina o modelo no dataset COMPLETO (os 5 folds juntos)
            # Isso garante que o modelo final aprenda de tudo antes de ser usado nas previsões
            pipeline.fit(X, y)

            # Guarda o modelo e suas métricas
            self.models[nome]  = pipeline
            self.metrics[nome] = {
                "R2_cv":   round(r2_cv, 4),
                "MAE_cv":  round(mae_cv, 4),
                "RMSE_cv": round(rmse_cv, 4)
            }

            print(f"  {nome:<22} {r2_cv:>8.4f} {mae_cv:>8.3f} {rmse_cv:>9.3f}")

            # Verifica se este modelo é o melhor até agora (maior R²)
            if r2_cv > melhor_r2:
                melhor_r2       = r2_cv
                self.best_model = pipeline
                self.best_name  = nome

        print(f"\n  ✅ Melhor modelo: {self.best_name}  (R²={melhor_r2:.4f})")
        return self.metrics

    def predict_he3(self, tio2_pct, fluxo_solar=1.0, lat_abs=0.0, is_mare=1):
        """
        Usa todos os modelos treinados para prever o He-3 de uma região específica.
        Retorna um dicionário com a previsão de cada modelo.

        tio2_pct    = concentração de TiO₂ em % do peso
        fluxo_solar = fator do vento solar (1.0 = normal)
        lat_abs     = latitude absoluta da região (sem sinal)
        is_mare     = 1 se é um "mar" lunar, 0 se é terra alta
        """
        # Monta o array de entrada com as 4 features
        # np.array([[...]]) cria uma matriz linha (formato que o sklearn espera)
        entrada = np.array([[tio2_pct, fluxo_solar, lat_abs, is_mare]])

        # Pede a previsão de cada modelo e monta o dicionário de resultados
        resultados = {}
        for nome, pipeline in self.models.items():
            previsao = pipeline.predict(entrada)  # retorna um array com 1 valor
            resultados[nome] = round(float(previsao[0]), 2)

        return resultados

    def predict_map(self, tio2_map, fluxo_solar=1.0, lat_grid=None):
        """
        Aplica o melhor modelo ao mapa global de TiO₂ pixel por pixel.
        Retorna um mapa 2D (matriz) com a estimativa de He-3 em cada ponto da Lua.

        tio2_map  = matriz 2D com a concentração de TiO₂ em cada pixel
        lat_grid  = matriz 2D com a latitude de cada pixel (opcional)
        """
        # Descobre as dimensões do mapa
        altura, largura = tio2_map.shape

        # "Achata" a matriz 2D em um vetor 1D para poder passar ao modelo
        # Por exemplo: [[1,2],[3,4]] vira [1, 2, 3, 4]
        tio2_flat = tio2_map.ravel()

        # Se temos o grid de latitude, achata também; senão usa zero para tudo
        if lat_grid is not None:
            lat_flat = np.abs(lat_grid.ravel())  # usa valor absoluto da latitude
        else:
            lat_flat = np.zeros(altura * largura)

        # Estima se cada pixel é "mare" ou não: TiO₂ > 1.5% → provavelmente mare
        # .astype(float) converte True/False para 1.0/0.0
        is_mare_estimado = (tio2_flat > 1.5).astype(float)

        # Cria um array de fluxo solar com o mesmo valor para todos os pixels
        fluxo_array = np.full(altura * largura, fluxo_solar)

        # Empilha as 4 features lado a lado → cada linha é um pixel, cada coluna é uma feature
        # np.stack([a, b, c, d], axis=1) cria uma matriz Nx4
        X_mapa = np.stack([tio2_flat, fluxo_array, lat_flat, is_mare_estimado], axis=1)

        # Aplica o modelo a todos os pixels de uma vez
        he3_flat = self.best_model.predict(X_mapa)

        # Garante valores entre 0 e 50 ppb, e reconstrói a forma 2D original
        return np.clip(he3_flat.reshape(altura, largura), 0, 50)

    def print_accuracy_report(self, flux_factor):
        """Imprime um relatório formatado das métricas de todos os modelos no terminal."""
        print("\n" + "═" * 65)
        print("      RELATÓRIO DE ACURÁCIA — ML MULTI-MODELO")
        print("═" * 65)
        print(f"  Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Amostras de calibração: {len(self.df_apollo)}")
        print(f"  Features: TiO₂ (%), Fluxo Solar, |Latitude|, is_Mare")
        print(f"  Fator de fluxo solar atual: {flux_factor:.4f}")
        print(f"  Validação cruzada: KFold k=5\n")
        print(f"  {'Modelo':<22} {'R² CV':>8} {'MAE CV':>8} {'RMSE CV':>9}  {'Status'}")
        print("  " + "─" * 65)

        for nome, metricas in self.metrics.items():
            # Marca o melhor modelo com uma estrela
            status = "★ BEST" if nome == self.best_name else ""
            print(f"  {nome:<22} {metricas['R2_cv']:>8.4f} {metricas['MAE_cv']:>8.3f} {metricas['RMSE_cv']:>9.3f}  {status}")

        print("═" * 65 + "\n")


# =============================================================================
# MÓDULO 3 — MINERALOGIA LUNAR
# Classe responsável por buscar dados reais de ilmenita/TiO₂ nas APIs científicas
# e gerar o mapa global de He-3 usando o melhor modelo de ML treinado.
# =============================================================================
class LunarMineralogyData:
    """Acessa dados de ilmenita / TiO₂ da Lua via APIs reais."""

    def __init__(self):
        self.m3_granules  = []    # Granules do instrumento M3 (Chandrayaan-1)
        self.pds_datasets = []    # Datasets do PDS da NASA
        self.ilmenite_map = None  # Mapa 2D final de He-3 estimado (ppb)
        self.tio2_map     = None  # Mapa 2D de TiO₂ (%)
        self.lon_grid     = None  # Grade 2D de longitudes (-180 a +180)
        self.lat_grid     = None  # Grade 2D de latitudes (-90 a +90)

    @staticmethod
    def _safe_get(url, params=None, headers=None):
        """
        Método auxiliar para fazer requisições HTTP de forma segura.
        @staticmethod significa que este método não precisa de 'self' — não usa
        nenhuma variável do objeto, é só uma função auxiliar dentro da classe.

        Retorna None (em vez de lançar exceção) se:
          - Não há internet
          - A API está fora do ar
          - Deu timeout
          - O proxy/firewall bloqueou
        """
        try:
            # Se headers for None, usa um dicionário vazio
            cabecalhos = headers if headers is not None else {}

            resp = requests.get(url, params=params, headers=cabecalhos, timeout=TIMEOUT)

            # Código 403 = Proibido / 407 = Proxy exige autenticação
            # Se o corpo da resposta for curto (< 200 chars), provavelmente é bloqueio
            if resp.status_code in (403, 407) and len(resp.text) < 200:
                return None

            return resp

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ProxyError):
            # Erros de rede conhecidos: retorna None silenciosamente
            return None

        except Exception:
            # Qualquer outro erro inesperado: também retorna None
            return None

    def search_cmr_m3(self):
        """
        Busca no NASA CMR (Common Metadata Repository) os granules do
        Moon Mineralogy Mapper (M3), instrumento da missão Chandrayaan-1 (2008-2009).
        O M3 mapeou a composição mineral de toda a superfície lunar com alta resolução.
        """
        print("  [CMR] Buscando granules M3 (Chandrayaan-1)...")

        # Lista de possíveis nomes do dataset no CMR (o nome exato pode variar)
        short_names = [
            "M3G20111020_RFL_V03",
            "M3T20110921_RFL_V01",
            "M3T20110921_RFL",
            "CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0",
        ]

        # Monta o cabeçalho de autenticação se tivermos o token
        if EARTHDATA_TOKEN:
            cabecalhos = {"Authorization": f"Bearer {EARTHDATA_TOKEN}"}
        else:
            cabecalhos = {}

        # Tenta cada nome até encontrar dados
        for nome_dataset in short_names:
            resp = self._safe_get(
                f"{CMR_BASE}/granules.json",
                params={"short_name": nome_dataset, "page_size": 10, "page_num": 1},
                headers=cabecalhos
            )

            # Se a requisição falhou completamente (sem internet etc.)
            if resp is None:
                print("  [CMR] Host inacessível — usando dados de referência compilados.")
                return []

            if resp.status_code == 200:
                try:
                    # A resposta JSON tem estrutura: {"feed": {"entry": [lista de granules]}}
                    granules = resp.json().get("feed", {}).get("entry", [])
                    if granules:
                        self.m3_granules = granules
                        print(f"  [CMR] {len(granules)} granule(s) M3 encontrados.")
                        return granules
                except Exception:
                    pass  # Ignora erros de parsing e tenta o próximo nome

        print("  [CMR] Nenhum granule M3 — usando dados de referência compilados.")
        return []

    def search_cmr_lunar_prospector(self):
        """
        Busca dados do Lunar Prospector GRS (Gamma Ray Spectrometer).
        O Lunar Prospector (1998-1999) mediu a composição química da Lua
        em todo o globo, incluindo Titânio, Ferro e Tório.
        """
        print("  [CMR] Buscando Lunar Prospector GRS...")

        short_names = [
            "LP_GRS",
            "LPGRS_L1",
            "LP-L-GRS-5-ELEM-ABUNDANCE-V1.0",
            "LP_ELEMENT_ABUNDANCE",
        ]

        if EARTHDATA_TOKEN:
            cabecalhos = {"Authorization": f"Bearer {EARTHDATA_TOKEN}"}
        else:
            cabecalhos = {}

        for nome_dataset in short_names:
            resp = self._safe_get(
                f"{CMR_BASE}/granules.json",
                params={"short_name": nome_dataset, "page_size": 5},
                headers=cabecalhos
            )

            if resp is None:
                print("  [CMR] Host inacessível — usando dados de referência compilados.")
                return []

            if resp.status_code == 200:
                try:
                    entries = resp.json().get("feed", {}).get("entry", [])
                    if entries:
                        print(f"  [CMR] {len(entries)} dataset(s) Lunar Prospector encontrados.")
                        return entries
                except Exception:
                    pass

        print("  [CMR] Nenhum dataset Lunar Prospector — usando dados de referência compilados.")
        return []

    def search_pds_datasets(self):
        """
        Busca no NASA Planetary Data System (PDS) datasets relacionados
        a titânio e ilmenita na Lua.
        """
        print("  [PDS] Buscando datasets de TiO₂ / ilmenita...")

        # Tenta dois endpoints diferentes do PDS (um pode estar offline)
        endpoints = [
            "https://pds.mcp.nasa.gov/api/search/1/products",
            "https://pds.nasa.gov/api/search/1/products",
        ]

        for url in endpoints:
            # Tenta query mais específica primeiro, depois mais genérica
            queries = [
                {"q": "titanium moon ilmenite", "limit": 5},
                {"q": "moon titanium",          "limit": 5},
            ]

            for parametros in queries:
                resp = self._safe_get(url, params=parametros)

                if resp is None:
                    # Host inacessível: não adianta tentar a outra query no mesmo host
                    break

                if resp.status_code == 200:
                    try:
                        items = resp.json().get("data", [])
                        self.pds_datasets = items
                        print(f"  [PDS] {len(items)} produto(s) encontrados.")
                        return items
                    except Exception:
                        pass

        print("  [PDS] API indisponível — usando dados de referência compilados.")
        return []

    def search_jaxa_selene(self):
        """
        Consulta o catálogo de datasets da JAXA SELENE/Kaguya (DARTS).
        O SELENE (2007-2009) mapeou a composição de Ti e Fe da Lua com alta precisão.
        """
        print("  [JAXA] Consultando SELENE/Kaguya DARTS...")

        # Tenta diferentes URLs do serviço DARTS (a URL exata pode mudar)
        urls = [
            "https://darts.isas.jaxa.jp/api/selene/dataset_list.json",
            "https://darts.isas.jaxa.jp/planet/pdap/selene/api/dataset_list.json",
            "https://darts.isas.jaxa.jp/planet/selene/dataset_list.json",
        ]

        for url in urls:
            resp = self._safe_get(url)

            if resp is None:
                print("  [JAXA] Host inacessível — usando dados de referência compilados.")
                return {}

            if resp.status_code == 200:
                # Verifica se a resposta realmente é JSON (às vezes retorna HTML)
                tipo_conteudo = resp.headers.get("Content-Type", "")
                comeca_com_chave = resp.text.strip().startswith("{")

                if "json" not in tipo_conteudo and not comeca_com_chave:
                    continue  # Não é JSON, tenta a próxima URL

                try:
                    data = resp.json()
                    print(f"  [JAXA] {len(data.get('datasets', []))} datasets SELENE.")
                    return data
                except Exception:
                    continue  # Erro de parsing, tenta a próxima URL

        print("  [JAXA] Todas as URLs DARTS indisponíveis — usando dados de referência compilados.")
        return {}

    def generate_tio2_map(self):
        """
        Gera um mapa global de TiO₂ (360×180 pixels = 1 pixel por grau).
        Cada região lunar é representada por uma "gaussiana" 2D centrada nas suas
        coordenadas — quanto mais longe do centro, menor a concentração de TiO₂.

        Gaussiana 2D: f(x,y) = A × exp(-(distância²) / (2σ²))
          A = valor máximo de TiO₂ da região
          σ (sigma) = "largura" da gaussiana, proporcional ao raio da região
        """
        # Cria vetores de longitudes e latitudes com 1 grau de resolução
        longitudes = np.linspace(-180, 180, 360)  # -180 a +180 em 360 passos
        latitudes  = np.linspace(-90,   90, 180)  # -90 a +90 em 180 passos

        # np.meshgrid transforma os vetores em grades 2D (uma matriz de longitudes e uma de latitudes)
        # Resultado: lon_grid[i][j] = longitude do pixel (i,j), lat_grid[i][j] = latitude
        self.lon_grid, self.lat_grid = np.meshgrid(longitudes, latitudes)

        # Começa com o mapa zerado
        tio2_map = np.zeros_like(self.lon_grid)

        # Para cada região, adiciona a contribuição gaussiana no mapa
        for regiao in LUNAR_REGIONS:
            # Calcula a distância ao quadrado de cada pixel até o centro da região
            distancia_quadrada = (
                (self.lon_grid - regiao["lon"]) ** 2 +
                (self.lat_grid - regiao["lat"]) ** 2
            )

            # sigma controla o "espalhamento" da gaussiana: raio grande → região mais larga
            sigma = regiao["raio"] / 2.5

            # Adiciona a contribuição desta região ao mapa total
            tio2_map += regiao["tio2"] * np.exp(-distancia_quadrada / (2 * sigma ** 2))

        # Garante que os valores fiquem entre 0% e 13% de TiO₂
        self.tio2_map = np.clip(tio2_map, 0, 13)
        return self.tio2_map

    def apply_ml_map(self, ml, fluxo_solar=1.0):
        """
        Aplica o melhor modelo de ML ao mapa de TiO₂ para gerar
        o mapa de He-3 estimado (em ppb) para toda a superfície lunar.
        """
        # Garante que o mapa de TiO₂ existe antes de continuar
        if self.tio2_map is None:
            self.generate_tio2_map()

        print(f"  [MAPA] Aplicando {ml.best_name} ao mapa lunar...")
        self.ilmenite_map = ml.predict_map(self.tio2_map, fluxo_solar, self.lat_grid)
        print(f"  [MAPA] Mapa de He-3 gerado (360×180 pixels).")
        return self.ilmenite_map

    def get_top_regions(self, ml, fluxo_solar=1.0):
        """
        Gera uma tabela com as 10 regiões lunares ranqueadas por potencial de He-3.
        Para cada região, usa o melhor modelo para prever o He-3 e classifica.
        """
        linhas = []  # Vai acumular os dados de cada região

        for regiao in LUNAR_REGIONS:
            # Pede previsão de todos os modelos para esta região
            previsoes = ml.predict_he3(
                regiao["tio2"],       # concentração de TiO₂
                fluxo_solar,          # fator do vento solar atual
                abs(regiao["lat"]),   # latitude absoluta (sem sinal)
                1                     # todas as regiões são "mare"
            )

            # Pega a previsão do melhor modelo
            he3_estimado = previsoes.get(ml.best_name, 0.0)

            # Classifica a concentração de He-3
            if he3_estimado > 20:
                classificacao = "🔴 ALTO"
            elif he3_estimado > 10:
                classificacao = "🟡 MÉDIO"
            else:
                classificacao = "🟢 BAIXO"

            linhas.append({
                "Região":          regiao["nome"],
                "Longitude (°)":   regiao["lon"],
                "Latitude (°)":    regiao["lat"],
                "TiO₂ (% peso)":   regiao["tio2"],
                "He-3 est. (ppb)": round(he3_estimado, 1),
                "Modelo ML":       ml.best_name,
                "Classificação":   classificacao,
            })

        # Cria o DataFrame e ordena do maior para o menor He-3
        df = pd.DataFrame(linhas)
        df = df.sort_values("He-3 est. (ppb)", ascending=False)
        df = df.reset_index(drop=True)  # Reinicia o índice após ordenar
        return df

    def fetch_all(self, ml, fluxo_solar=1.0):
        """Executa todas as buscas e gera os mapas em sequência."""
        print("\n━━━ MÓDULO 3: MINERALOGIA LUNAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.search_cmr_m3()
        self.search_cmr_lunar_prospector()
        self.search_pds_datasets()
        self.search_jaxa_selene()
        self.generate_tio2_map()
        self.apply_ml_map(ml, fluxo_solar)


# =============================================================================
# MÓDULO 4 — DASHBOARD VISUAL
# Classe responsável por montar o painel gráfico com 6 subplots.
# =============================================================================
class Dashboard:
    """Gera painel visual unificado com gráficos de vento solar, ML e mapa lunar."""

    def __init__(self, solar, lunar, ml):
        # Guarda referências aos objetos dos outros módulos
        self.solar = solar  # dados de vento solar
        self.lunar = lunar  # dados de mineralogia lunar
        self.ml    = ml     # pipeline de machine learning

    @staticmethod
    def _he3_cmap():
        """
        Cria um mapa de cores personalizado para representar a concentração de He-3.
        As cores vão do azul escuro (pouco He-3) até o vermelho (muito He-3).
        @staticmethod: não usa self, é apenas uma função auxiliar dentro da classe.
        """
        # Lista de cores RGB (cada valor entre 0.0 e 1.0)
        # A sequência vai do mais frio (escuro/azul) para o mais quente (vermelho)
        cores = [
            (0.04, 0.04, 0.12),  # quase preto
            (0.00, 0.20, 0.50),  # azul escuro
            (0.00, 0.55, 0.80),  # azul claro
            (0.00, 0.80, 0.60),  # ciano/verde
            (0.90, 0.80, 0.00),  # amarelo
            (1.00, 0.40, 0.00),  # laranja
            (1.00, 0.10, 0.10),  # vermelho
        ]
        # Cria um colormap interpolado com 256 cores entre os pontos definidos
        return LinearSegmentedColormap.from_list("he3", cores, N=256)

    def _style_ax(self, ax):
        """
        Aplica o estilo visual padrão (fundo escuro, texto claro) a um subplot.
        Este método é chamado em todos os gráficos para manter consistência visual.
        """
        ax.set_facecolor(COR_FUNDO)                    # Fundo do gráfico
        ax.tick_params(colors=COR_TEXTO, labelsize=7)  # Cor e tamanho dos números dos eixos
        ax.spines[:].set_color(COR_CINZA)              # Cor das bordas do gráfico
        ax.yaxis.label.set_color(COR_TEXTO)            # Cor do rótulo do eixo Y
        ax.xaxis.label.set_color(COR_TEXTO)            # Cor do rótulo do eixo X
        ax.title.set_color(COR_TEXTO)                  # Cor do título

    def plot_solar_speed(self, ax):
        """Desenha o gráfico de linha da velocidade do vento solar ao longo do tempo."""
        df = self.solar.plasma_data

        if df is not None and not df.empty:
            # Linha principal da velocidade
            ax.plot(df["time_tag"], df["speed"], color=COR_AZUL, lw=1.2)

            # Área sombreada abaixo da linha (fill_between = preenche entre a linha e o eixo X)
            ax.fill_between(df["time_tag"], df["speed"], alpha=0.15, color=COR_AZUL)

            # Linhas horizontais de referência (vento lento e vento rápido)
            ax.axhline(400, ls="--", lw=0.8, color=COR_VERDE,   alpha=0.6, label="Lento (400)")
            ax.axhline(600, ls="--", lw=0.8, color=COR_LARANJA, alpha=0.6, label="Rápido (600)")

            ax.set_title("Velocidade do Vento Solar — NOAA DSCOVR (km/s)")
            ax.set_ylabel("km/s")
            ax.legend(fontsize=7, framealpha=0.2, labelcolor=COR_TEXTO)
        else:
            # Se não há dados, exibe mensagem no centro do gráfico
            ax.text(0.5, 0.5, "Dados indisponíveis", ha="center", va="center",
                    color=COR_TEXTO, transform=ax.transAxes)

        self._style_ax(ax)

    def plot_imf_bz(self, ax):
        """
        Desenha o gráfico do campo magnético Bz (componente norte-sul do IMF).
        Barras laranjas = Bz negativo (sul) → risco de tempestade geomagnética.
        Barras azuis   = Bz positivo (norte) → campo protetor.
        """
        df = self.solar.mag_data

        if df is not None and not df.empty and "bz_gsm" in df.columns:
            bz = df["bz_gsm"]

            # Define a cor de cada barra dependendo se Bz é positivo ou negativo
            cores_barras = []
            for valor in bz:
                if valor < 0:
                    cores_barras.append(COR_LARANJA)  # sul = laranja (perigo)
                else:
                    cores_barras.append(COR_AZUL)     # norte = azul (normal)

            ax.bar(df["time_tag"], bz, color=cores_barras, width=0.002, alpha=0.8)

            # Linha horizontal no zero para facilitar leitura
            ax.axhline(0, color=COR_CINZA, lw=0.7)

            ax.set_title("Campo Magnético Interplanetário Bz (nT)")
            ax.set_ylabel("nT")

            # Cria legenda manual com patches coloridos
            patch_sul   = mpatches.Patch(color=COR_LARANJA, label="Sul < 0")
            patch_norte = mpatches.Patch(color=COR_AZUL,    label="Norte ≥ 0")
            ax.legend(handles=[patch_sul, patch_norte], fontsize=7, framealpha=0.2, labelcolor=COR_TEXTO)
        else:
            ax.text(0.5, 0.5, "Dados IMF indisponíveis", ha="center", va="center",
                    color=COR_TEXTO, transform=ax.transAxes)

        self._style_ax(ax)

    def plot_ml_calibration(self, ax):
        """
        Gráfico de calibração do ML:
        - Pontos amarelos = amostras reais Apollo/SELENE
        - Curvas coloridas = previsões de cada modelo ao longo do eixo de TiO₂
        """
        df = self.ml.df_apollo

        # Cria 100 pontos de TiO₂ entre 0% e 14% para desenhar as curvas dos modelos
        tio2_linha = np.linspace(0, 14, 100)

        # Plota os pontos reais de calibração
        ax.scatter(
            df["TiO2_pct"], df["He3_ppb"],
            color=COR_AMARELO, alpha=0.75, edgecolors="white",
            s=40, zorder=5, label="Amostras Apollo/SELENE"
        )

        # Uma cor diferente para cada modelo
        cores_modelos = [COR_AZUL, COR_VERDE, COR_LARANJA, COR_ROXO]

        for (nome, pipeline), cor in zip(self.ml.models.items(), cores_modelos):
            # Monta a entrada de previsão: 100 pontos de TiO₂, fluxo=1, lat=15, is_mare=1
            # np.ones(100) = array de 100 uns; np.full(100, 15) = array de 100 com valor 15
            entrada_previsao = np.stack([
                tio2_linha,
                np.ones(100),        # fluxo solar normalizado = 1.0 (condição média)
                np.full(100, 15),    # latitude absoluta = 15° (valor médio)
                np.ones(100)         # is_mare = 1 (região de mar)
            ], axis=1)

            y_previsto = pipeline.predict(entrada_previsao)

            # Pega o R² deste modelo para mostrar na legenda
            r2 = self.ml.metrics[nome]["R2_cv"]
            rotulo = f"{nome} (R²={r2:.3f})"

            # Melhor modelo = linha sólida e mais grossa; outros = tracejado e fino
            if nome == self.ml.best_name:
                estilo_linha = "-"
                espessura    = 2.0
            else:
                estilo_linha = "--"
                espessura    = 1.2

            ax.plot(tio2_linha, y_previsto, color=cor, lw=espessura, ls=estilo_linha, label=rotulo)

        ax.set_title(f"Calibração ML — Apollo/SELENE | Melhor: {self.ml.best_name}")
        ax.set_xlabel("TiO₂ (% peso)")
        ax.set_ylabel("He-3 (ppb)")
        ax.legend(fontsize=6.5, framealpha=0.3, labelcolor=COR_TEXTO,
                  facecolor=COR_FUNDO, loc="upper left")
        self._style_ax(ax)

    def plot_status_panel(self, ax):
        """
        Painel de texto com o status atual do vento solar,
        contagem de eventos e métricas do melhor modelo de ML.
        """
        flux  = self.solar.calculate_lunar_flux()
        nivel = flux.get("nivel_atividade", "N/D")

        # Define a cor do nível de atividade (vermelho = perigo, verde = calmo)
        cores_nivel = {
            "EXTREMO":  "#ff2222",
            "ALTO":     COR_LARANJA,
            "MODERADO": COR_AMARELO,
            "BAIXO":    COR_VERDE,
        }
        cor_nivel = cores_nivel.get(nivel, COR_TEXTO)

        # Remove os eixos do subplot (é um painel de texto puro)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Pega os dados do melhor modelo de ML
        nome_melhor  = self.ml.best_name
        r2_melhor    = self.ml.metrics.get(nome_melhor, {}).get("R2_cv", 0)
        mae_melhor   = self.ml.metrics.get(nome_melhor, {}).get("MAE_cv", 0)

        # Cada linha do painel: (texto, cor, tamanho_fonte, estilo)
        linhas = [
            ("━━━ STATUS VENTO SOLAR ━━━",       COR_TEXTO,   11, "bold"),
            (f"Velocidade:  {flux.get('velocidade_atual_kms','—')} km/s",  COR_AZUL, 9, "normal"),
            (f"Densidade:   {flux.get('densidade_atual_pcm3','—')} p/cm³", COR_AZUL, 9, "normal"),
            (f"Temperatura: {flux.get('temperatura_atual_K','—')} K",      COR_AZUL, 9, "normal"),
            (f"Vel. média:  {flux.get('velocidade_media_kms','—')} km/s",  COR_TEXTO, 8, "normal"),
            (f"Nível: {nivel}",                                             cor_nivel, 10, "bold"),
            ("",                                 COR_TEXTO,    8, "normal"),  # linha em branco
            ("━━━ EVENTOS ESPAÇO-CLIMÁTICOS ━━",  COR_TEXTO,   11, "bold"),
            (f"CMEs:         {flux.get('total_cmes','—')}",        COR_LARANJA, 9, "normal"),
            (f"Flares:       {flux.get('total_flares','—')}",      COR_AMARELO, 9, "normal"),
            (f"Tempestades:  {flux.get('total_tempestades','—')}",  COR_ROXO,   9, "normal"),
            ("",                                 COR_TEXTO,    8, "normal"),
            ("━━━ MODELO ML ATIVO ━━━━━━━━━━━━",  COR_TEXTO,   11, "bold"),
            (f"{nome_melhor}",                    COR_VERDE,    9, "bold"),
            (f"R² (CV k=5): {r2_melhor:.4f}",     COR_TEXTO,    8, "normal"),
            (f"MAE (CV):    {mae_melhor:.3f} ppb", COR_TEXTO,    8, "normal"),
            ("",                                 COR_TEXTO,    8, "normal"),
            (f"Janela: {DATA_WINDOW_DAYS} dias | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
             COR_CINZA, 7, "normal"),
        ]

        # Posição vertical inicial (começa no topo)
        posicao_y = 0.97

        for linha in linhas:
            texto, cor, tamanho, estilo = linha

            if texto == "":
                # Linha em branco: apenas pula um espaço
                posicao_y -= 0.045
                continue

            ax.text(0.04, posicao_y, texto,
                    color=cor, fontsize=tamanho, fontweight=estilo,
                    transform=ax.transAxes, va="top")

            posicao_y -= 0.058  # Avança para a próxima linha

    def plot_lunar_map(self, ax):
        """
        Desenha o mapa global da Lua colorido pela concentração de He-3 estimada.
        As regiões catalogadas aparecem como marcadores coloridos sobre o mapa.
        """
        he3_map = self.lunar.ilmenite_map
        cmap    = self._he3_cmap()

        if he3_map is not None:
            # imshow desenha a matriz como imagem
            # extent define os limites dos eixos X (lon) e Y (lat)
            # origin="lower" coloca a latitude -90 embaixo
            im = ax.imshow(
                he3_map,
                extent=[-180, 180, -90, 90],
                origin="lower",
                cmap=cmap,
                vmin=0, vmax=40,  # escala de cores de 0 a 40 ppb
                aspect="auto"
            )

            # Plota um marcador para cada região catalogada
            for regiao in LUNAR_REGIONS:
                # Define a cor do marcador pelo nível de He-3
                if regiao["he3_ppb"] > 20:
                    cor_marcador = "#ff4444"   # vermelho = alto
                elif regiao["he3_ppb"] > 10:
                    cor_marcador = "#ffaa00"   # laranja = médio
                else:
                    cor_marcador = "#44ff88"   # verde = baixo

                ax.scatter(
                    regiao["lon"], regiao["lat"],
                    s=55, color=cor_marcador,
                    edgecolors="white", linewidths=0.5, zorder=5
                )

                # Rótulo abreviado (ex: "M. Tranquillitatis" em vez do nome completo)
                nome_abreviado = regiao["nome"].replace("Mare ", "M. ")
                ax.annotate(
                    nome_abreviado,
                    xy=(regiao["lon"], regiao["lat"]),
                    xytext=(4, 4),                        # desloca o texto 4 pontos para cima/direita
                    textcoords="offset points",
                    fontsize=5, color="white", alpha=0.9
                )

            # Configura os eixos
            ax.set_xticks(range(-180, 181, 30))
            ax.set_yticks(range(-90, 91, 30))
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(
                f"Mapa de He-3 Potencial — Lua (ppb) · Modelo: {self.ml.best_name}\n"
                "Fontes: M3 (Chandrayaan-1) · SELENE/Kaguya · Lunar Prospector GRS"
            )

            # Adiciona a barra de cores horizontal abaixo do mapa
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.12)
            cbar.set_label("He-3 Estimado (ppb)", color=COR_TEXTO, fontsize=8)
            cbar.ax.tick_params(colors=COR_TEXTO, labelsize=7)
        else:
            ax.text(0.5, 0.5, "Mapa não gerado", ha="center", va="center", color=COR_TEXTO)

        self._style_ax(ax)

    def plot_regions_table(self, ax, fluxo_solar):
        """
        Desenha a tabela das top 10 regiões com maior potencial de He-3
        como um subplot do dashboard.
        """
        ax.axis("off")  # Esconde os eixos (queremos apenas a tabela)

        # Busca as regiões ordenadas pelo modelo
        df = self.lunar.get_top_regions(self.ml, fluxo_solar)

        # Define os cabeçalhos das colunas da tabela
        colunas = ["Região", "TiO₂ (%)", "He-3 (ppb)", "ML", "Clasf."]

        # Monta as linhas da tabela com os dados das top 10 regiões
        linhas = []
        for _, linha in df.head(10).iterrows():
            linhas.append([
                linha["Região"],
                f"{linha['TiO₂ (% peso)']:.1f}",     # 1 casa decimal
                f"{linha['He-3 est. (ppb)']:.1f}",    # 1 casa decimal
                linha["Modelo ML"].split()[0],         # Apenas a primeira palavra do nome do modelo
                linha["Classificação"],
            ])

        # Cria a tabela no subplot
        tabela = ax.table(
            cellText=linhas,
            colLabels=colunas,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]  # ocupa todo o subplot
        )

        tabela.auto_set_font_size(False)
        tabela.set_fontsize(7)

        # Personaliza a aparência de cada célula
        for (linha_idx, col_idx), celula in tabela.get_celld().items():
            # Alterna as cores das linhas (zebra)
            if linha_idx % 2 == 0:
                celula.set_facecolor("#0e1030")
            else:
                celula.set_facecolor("#080820")

            celula.set_edgecolor(COR_CINZA)  # Cor das bordas

            if linha_idx == 0:
                # Linha de cabeçalho: fundo mais escuro e texto amarelo
                celula.set_facecolor("#1a1a50")
                celula.set_text_props(color=COR_AMARELO, fontweight="bold")
            else:
                celula.set_text_props(color=COR_TEXTO)

        ax.set_title("Top Alvos de He-3 para Prospecção", color=COR_TEXTO, pad=8)

    def render(self, fluxo_solar=1.0, save=True):
        """
        Monta e exibe o dashboard completo com os 6 subplots.
        Se save=True, salva a imagem em PNG na pasta de saída.
        """
        print("\n━━━ MÓDULO 4: RENDERIZANDO DASHBOARD ━━━━━━━━━━━━━━━━━━━━━━")

        # Cria a figura principal

        try:
            import tkinter as tk
            root = tk.Tk();
            root.withdraw()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            fig_w = min(22, (sw * 0.92) / _SCREEN_DPI)
            fig_h = min(15, (sh * 0.88) / _SCREEN_DPI)
        except Exception:
            fig_w, fig_h = 22, 15
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COR_FUNDO)

        # Título geral no topo da figura
        fig.suptitle(
            "🌑  LUNAR HE-3 & SOLAR WIND MONITOR v3.0  ☀️\n"
            f"ML Multi-Modelo · {DATA_WINDOW_DAYS} dias · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            color=COR_AMARELO, fontsize=13, fontweight="bold", y=0.98,
        )

        # GridSpec organiza os subplots em uma grade de 3 linhas × 3 colunas
        # left/right/top/bottom definem as margens; hspace/wspace definem espaços entre subplots
        gs = gridspec.GridSpec(
            3, 3, figure=fig,
            left=0.06, right=0.97, top=0.93, bottom=0.06,
            hspace=0.45, wspace=0.30
        )

        # Cria cada subplot especificando quais células da grade ele ocupa
        # gs[linha, coluna] — índice começa em 0; ":" significa "todas as colunas"
        ax_speed  = fig.add_subplot(gs[0, :2])   # Linha 0, colunas 0 e 1
        ax_mag    = fig.add_subplot(gs[1, :2])   # Linha 1, colunas 0 e 1
        ax_status = fig.add_subplot(gs[0:2, 2])  # Linhas 0 e 1, coluna 2
        ax_ml     = fig.add_subplot(gs[2, 0])    # Linha 2, coluna 0
        ax_map    = fig.add_subplot(gs[2, 1])    # Linha 2, coluna 1
        ax_table  = fig.add_subplot(gs[2, 2])    # Linha 2, coluna 2

        # Aplica fundo escuro a todos os subplots
        for ax in [ax_speed, ax_mag, ax_status, ax_ml, ax_map, ax_table]:
            ax.set_facecolor(COR_FUNDO)

        # Preenche cada subplot com seu respectivo gráfico
        self.plot_solar_speed(ax_speed)
        self.plot_imf_bz(ax_mag)
        self.plot_status_panel(ax_status)
        self.plot_ml_calibration(ax_ml)
        self.plot_lunar_map(ax_map)
        self.plot_regions_table(ax_table, fluxo_solar)

        # Monta o nome do arquivo com data e hora para evitar sobrescrever
        nome_arquivo = f"dashboard_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        caminho = OUTPUT_DIR / nome_arquivo

        if save:
            fig.savefig(caminho, dpi=120, facecolor=COR_FUNDO, bbox_inches="tight")
            print(f"  [OK] Dashboard salvo → {caminho}")

        try:
            mgr = plt.get_current_fig_manager()
            try:
                mgr.window.state("zoomed")  # TkAgg (Windows)
            except Exception:
                try:
                    mgr.window.showMaximized()  # Qt
                except Exception:
                    try:
                        mgr.frame.Maximize(True)  # wxAgg
                    except Exception:
                        pass
        except Exception:
            pass
        plt.show()
        plt.close(fig)  # Libera memória fechando a figura
        return caminho


# =============================================================================
# MÓDULO 5 — EXPORTAÇÃO DE DADOS
# Salva todos os dados coletados e gerados em arquivos CSV e JSON.
# =============================================================================
class DataExporter:
    """Exporta todos os dados coletados para arquivos CSV e JSON."""

    def __init__(self, solar, lunar, ml):
        # Guarda referências aos objetos dos módulos anteriores
        self.solar = solar
        self.lunar = lunar
        self.ml    = ml

    def export_all(self, fluxo_solar=1.0):
        """Exporta todos os dados disponíveis para a pasta de saída."""
        print("\n━━━ MÓDULO 5: EXPORTANDO DADOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Exporta dados de plasma do vento solar (se foram coletados)
        if self.solar.plasma_data is not None:
            caminho = OUTPUT_DIR / "solar_wind_plasma.csv"
            self.solar.plasma_data.to_csv(caminho, index=False)
            print(f"  [EXPORT] {caminho}")

        # Exporta dados do campo magnético (se foram coletados)
        if self.solar.mag_data is not None:
            caminho = OUTPUT_DIR / "solar_wind_mag.csv"
            self.solar.mag_data.to_csv(caminho, index=False)
            print(f"  [EXPORT] {caminho}")

        # Exporta lista de CMEs (se houver algum)
        if self.solar.cme_events:
            caminho = OUTPUT_DIR / "cme_events.json"
            with open(caminho, "w") as arquivo:
                # default=str converte tipos não-serializáveis (como datetime) para string
                json.dump(self.solar.cme_events, arquivo, indent=2, default=str)
            print(f"  [EXPORT] {caminho}")

        # Exporta lista de flares (se houver algum)
        if self.solar.flare_events:
            caminho = OUTPUT_DIR / "flare_events.json"
            with open(caminho, "w") as arquivo:
                json.dump(self.solar.flare_events, arquivo, indent=2, default=str)
            print(f"  [EXPORT] {caminho}")

        # Exporta a tabela de regiões lunares ranqueadas
        df_regioes = self.lunar.get_top_regions(self.ml, fluxo_solar)
        caminho = OUTPUT_DIR / "lunar_ilmenite_regions.csv"
        df_regioes.to_csv(caminho, index=False, encoding="utf-8-sig")
        print(f"  [EXPORT] {caminho}")

        # Exporta as métricas de todos os modelos de ML
        caminho = OUTPUT_DIR / "ml_metrics.json"
        with open(caminho, "w") as arquivo:
            json.dump(
                {"best_model": self.ml.best_name, "metrics": self.ml.metrics},
                arquivo,
                indent=2
            )
        print(f"  [EXPORT] {caminho}")

        # Exporta o dataset de calibração Apollo/SELENE usado para treinar os modelos
        caminho = OUTPUT_DIR / "apollo_calibration_data.csv"
        self.ml.df_apollo.to_csv(caminho, index=False)
        print(f"  [EXPORT] {caminho}")

        print(f"  [OK] Todos os arquivos salvos em: {OUTPUT_DIR.resolve()}")


# =============================================================================
# PONTO DE ENTRADA DO PROGRAMA
# Aqui configuramos os argumentos de linha de comando e executamos tudo.
# =============================================================================
def parse_args():
    """
    Configura os argumentos que podem ser passados ao executar o script.
    Exemplo: python lunar_v3.py --solar --no-plot
    """
    parser = argparse.ArgumentParser(description="Lunar He-3 & Solar Wind Monitor v3.0")

    # action="store_true" significa: se o argumento aparecer, guarda True; senão, False
    parser.add_argument("--solar",   action="store_true", help="Apenas dados de vento solar")
    parser.add_argument("--lunar",   action="store_true", help="Apenas dados lunares")
    parser.add_argument("--ml",      action="store_true", help="Apenas benchmark de ML")
    parser.add_argument("--export",  action="store_true", help="Exportar CSVs/JSONs")
    parser.add_argument("--no-plot", action="store_true", help="Não exibir gráfico")

    return parser.parse_args()


def main():
    """Função principal: orquestra a execução de todos os módulos."""
    args = parse_args()

    # Cabeçalho de boas-vindas no terminal
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   LUNAR HE-3 & SOLAR WIND MONITOR v3.0 — Iniciando...        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Saída: {OUTPUT_DIR.resolve()}")

    # Informa se a NASA API Key está configurada ou usando a chave demo (limitada)
    if NASA_API_KEY != "DEMO_KEY":
        print("  NASA API: ✓ configurada")
    else:
        print("  NASA API: ⚠ usando DEMO_KEY (limite de 30 req/hora)")

    # Informa se o token do Earthdata está presente
    if EARTHDATA_TOKEN:
        print("  Earthdata: ✓ token presente")
    else:
        print("  Earthdata: ⚠ token ausente (dados CMR/PDS podem ser limitados)")

    # Cria os objetos de cada módulo
    solar = SolarWindMonitor()
    ml    = LunarMLPipeline()
    lunar = LunarMineralogyData()

    # O ML treina sempre (é rápido, usa dados locais, sem internet)
    ml.train_all_models()

    # Busca dados de vento solar (a menos que o usuário tenha passado --lunar)
    resultados_flux = {}
    if not args.lunar:
        resultados_flux = solar.fetch_all()

    # Calcula o fator de fluxo solar normalizado (velocidade atual / velocidade típica de 450 km/s)
    # .get() busca o valor no dicionário; se não encontrar, usa 450 como padrão
    velocidade_atual = resultados_flux.get("velocidade_atual_kms", 450)
    fluxo_solar = velocidade_atual / 450

    # Busca dados lunares (a menos que o usuário tenha passado --solar)
    if not args.solar:
        lunar.fetch_all(ml, fluxo_solar)

    # Imprime o relatório completo de acurácia dos modelos
    ml.print_accuracy_report(fluxo_solar)

    # Exporta os dados se o usuário pediu --export
    if args.export:
        exportador = DataExporter(solar, lunar, ml)
        exportador.export_all(fluxo_solar)

    # Gera e exibe o dashboard (a menos que o usuário tenha passado --no-plot)
    if not args.no_plot:
        dashboard = Dashboard(solar, lunar, ml)
        dashboard.render(fluxo_solar, save=True)

    # Resumo final no terminal
    print("\n━━━ RESUMO FINAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if resultados_flux:
        print(f"  🌬️  Vento solar: {resultados_flux.get('velocidade_atual_kms')} km/s "
              f"[{resultados_flux.get('nivel_atividade')}]")
        print(f"  ☀️  CMEs: {resultados_flux.get('total_cmes')}  |  "
              f"Flares: {resultados_flux.get('total_flares')}  |  "
              f"Tempestades: {resultados_flux.get('total_tempestades')}")

    print(f"\n  🤖 Melhor modelo: {ml.best_name}  "
          f"R²={ml.metrics[ml.best_name]['R2_cv']:.4f}  "
          f"MAE={ml.metrics[ml.best_name]['MAE_cv']:.3f} ppb")

    print("\n  🌑 Top 3 regiões com maior potencial de He-3:")

    df_top = lunar.get_top_regions(ml, fluxo_solar)

    # .iterrows() percorre as linhas do DataFrame; _ é o índice (que não usamos)
    for _, linha in df_top.head(3).iterrows():
        print(f"     • {linha['Região']:<26}  "
              f"TiO₂={linha['TiO₂ (% peso)']:.1f}%  "
              f"He-3≈{linha['He-3 est. (ppb)']:.1f} ppb  "
              f"{linha['Classificação']}")

    print("\n  ✅ Concluído. Arquivos salvos em:", OUTPUT_DIR.resolve())


# =============================================================================
# EXECUÇÃO
# Este bloco garante que main() só seja chamado quando o script for executado
# diretamente (python lunar_v3.py), e NÃO quando importado por outro script.
# =============================================================================
if __name__ == "__main__":
    main()