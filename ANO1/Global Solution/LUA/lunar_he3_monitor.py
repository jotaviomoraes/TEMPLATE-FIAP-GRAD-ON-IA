#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         LUNAR HE-3 & SOLAR WIND MONITOR  — v1.0                 ║
║                                                                  ║
║  Fontes de dados:                                                ║
║  • NOAA DSCOVR   → Vento solar em tempo real                     ║
║  • NASA DONKI    → CME, Flares, Tempestades geomagnéticas        ║
║  • NASA CMR/PDS  → M3 Chandrayaan-1 (Ilmenita lunar)            ║
║  • JAXA SELENE   → Composição Ti/Fe da Lua                       ║
║  • USGS          → Mapas TiO₂ (proxy Hélio-3)                   ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
    python lunar_he3_monitor.py              # Dashboard completo
    python lunar_he3_monitor.py --solar      # Apenas vento solar
    python lunar_he3_monitor.py --lunar      # Apenas mapa lunar
    python lunar_he3_monitor.py --export     # Exporta dados CSV
"""

import os
import sys
import json
import time
import argparse
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─────────────────────────────────────────────
#  Tenta carregar .env; se não existir, usa os.environ
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv opcional

# ══════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════
NASA_API_KEY     = os.getenv("NASA_API_KEY")
EARTHDATA_TOKEN  = os.getenv("EARTHDATA_TOKEN", "")
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "./outputs"))
DATA_WINDOW_DAYS = int(os.getenv("DATA_WINDOW_DAYS", "7"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# URLs base
NOAA_BASE   = "https://services.swpc.noaa.gov"
DONKI_BASE  = "https://api.nasa.gov/DONKI"
CMR_BASE    = "https://cmr.earthdata.nasa.gov/search"
PDS_BASE    = "https://pds.nasa.gov/api/search/1"

# Cabeçalhos padrão
EARTHDATA_HEADERS = {
    "Authorization": f"Bearer {EARTHDATA_TOKEN}",
    "Accept": "application/json",
}

# Timeout padrão para requests
TIMEOUT = 30

# ══════════════════════════════════════════════════════════════════
#  REGIÕES LUNARES COM ALTA ILMENITA / POTENCIAL HE-3
#  (dados compilados de Clementine, Kaguya/SELENE e Lunar Prospector)
# ══════════════════════════════════════════════════════════════════
LUNAR_REGIONS = [
    # nome, lon_centro, lat_centro, raio_graus, TiO2_%, estimativa_He3_ppb
    {"nome": "Mare Tranquillitatis",  "lon":  31.4, "lat":  8.5,  "raio": 6.0,  "tio2": 11.8, "he3_ppb": 28.0},
    {"nome": "Mare Imbrium",          "lon": -14.6, "lat": 32.8,  "raio": 7.5,  "tio2":  5.6, "he3_ppb": 13.0},
    {"nome": "Mare Serenitatis",      "lon":  17.5, "lat": 26.0,  "raio": 5.5,  "tio2":  7.2, "he3_ppb": 17.0},
    {"nome": "Mare Crisium",          "lon":  59.1, "lat": 17.0,  "raio": 4.5,  "tio2":  5.1, "he3_ppb": 12.0},
    {"nome": "Oceanus Procellarum N", "lon": -49.8, "lat": 23.0,  "raio": 8.0,  "tio2":  3.8, "he3_ppb":  9.0},
    {"nome": "Oceanus Procellarum S", "lon": -44.2, "lat":  0.0,  "raio": 7.0,  "tio2":  3.2, "he3_ppb":  8.0},
    {"nome": "Mare Fecunditatis",     "lon":  51.3, "lat":  -4.5, "raio": 5.0,  "tio2":  3.5, "he3_ppb":  8.5},
    {"nome": "Mare Nectaris",         "lon":  34.6, "lat": -15.0, "raio": 3.0,  "tio2":  3.0, "he3_ppb":  7.0},
    {"nome": "Mare Humorum",          "lon": -38.7, "lat": -24.0, "raio": 3.5,  "tio2":  4.0, "he3_ppb":  9.5},
    {"nome": "Mare Frigoris",         "lon":  -1.4, "lat":  56.0, "raio": 4.5,  "tio2":  2.1, "he3_ppb":  5.0},
]

# Cores do projeto
COR_FUNDO   = "#0a0a1a"
COR_TEXTO   = "#e0e8ff"
COR_AZUL    = "#3a8dde"
COR_LARANJA = "#ff6b35"
COR_VERDE   = "#39d353"
COR_ROXO    = "#b07aff"
COR_AMARELO = "#ffd700"
COR_CINZA   = "#555577"


# ══════════════════════════════════════════════════════════════════
#  MÓDULO 1 — VENTO SOLAR (NOAA + NASA DONKI)
# ══════════════════════════════════════════════════════════════════
class SolarWindMonitor:
    """Captura e analisa dados de vento solar de NOAA e NASA DONKI."""

    def __init__(self):
        self.plasma_data  = None
        self.mag_data     = None
        self.cme_events   = []
        self.flare_events = []
        self.storm_events = []

    # ── NOAA ──────────────────────────────────────────────────────

    def fetch_noaa_plasma(self) -> pd.DataFrame | None:
        """
        Busca dados de plasma do vento solar (últimos 7 dias).
        Campos: time, densidade (p/cm³), velocidade (km/s), temperatura (K)
        """
        url = f"{NOAA_BASE}/products/solar-wind/plasma-7-day.json"
        print("  [NOAA] Buscando plasma do vento solar...")
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            # primeira linha é cabeçalho
            cols = raw[0]
            data = raw[1:]
            df = pd.DataFrame(data, columns=cols)
            df["time_tag"] = pd.to_datetime(df["time_tag"])
            for c in ["density", "speed", "temperature"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["speed"])
            self.plasma_data = df
            print(f"  [NOAA] {len(df)} registros de plasma obtidos.")
            return df
        except Exception as e:
            print(f"  [NOAA] Erro ao buscar plasma: {e}")
            return None

    def fetch_noaa_magnetic(self) -> pd.DataFrame | None:
        """
        Busca dados do campo magnético interplanetário (IMF).
        Campos: time, Bx, By, Bz, Bt (nT), latitude, longitude
        """
        url = f"{NOAA_BASE}/products/solar-wind/mag-7-day.json"
        print("  [NOAA] Buscando campo magnético (IMF)...")
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            cols = raw[0]
            data = raw[1:]
            df = pd.DataFrame(data, columns=cols)
            df["time_tag"] = pd.to_datetime(df["time_tag"])
            for c in ["bx_gsm", "by_gsm", "bz_gsm", "bt"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            self.mag_data = df
            print(f"  [NOAA] {len(df)} registros IMF obtidos.")
            return df
        except Exception as e:
            print(f"  [NOAA] Erro ao buscar IMF: {e}")
            return None

    def fetch_noaa_kp_index(self) -> list:
        """Busca índice Kp (atividade geomagnética) dos últimos 7 dias."""
        url = f"{NOAA_BASE}/products/noaa-planetary-k-index-forecast.json"
        print("  [NOAA] Buscando índice Kp...")
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            # formato: [[time, Kp, A, Station], ...]
            cols  = raw[0]
            data  = raw[1:]
            df = pd.DataFrame(data, columns=cols)
            df["time_tag"] = pd.to_datetime(df["time_tag"])
            df["kp"]       = pd.to_numeric(df["kp"], errors="coerce")
            return df
        except Exception as e:
            print(f"  [NOAA] Erro ao buscar Kp: {e}")
            return pd.DataFrame()

    # ── NASA DONKI ─────────────────────────────────────────────────

    def _donki_get(self, endpoint: str, params: dict) -> list:
        """Requisição genérica ao DONKI."""
        params["api_key"] = NASA_API_KEY
        try:
            resp = requests.get(f"{DONKI_BASE}/{endpoint}", params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"  [DONKI] Erro em {endpoint}: {e}")
            return []

    def fetch_donki_cme(self) -> list:
        """Ejections de massa coronal (CME) nos últimos N dias."""
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=DATA_WINDOW_DAYS)
        print("  [DONKI] Buscando CMEs...")
        events = self._donki_get("CME", {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate":   end.strftime("%Y-%m-%d"),
        })
        self.cme_events = events
        print(f"  [DONKI] {len(events)} CME(s) encontrados.")
        return events

    def fetch_donki_flares(self) -> list:
        """Flares solares (X, M, C, B) nos últimos N dias."""
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=DATA_WINDOW_DAYS)
        print("  [DONKI] Buscando Flares...")
        events = self._donki_get("FLR", {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate":   end.strftime("%Y-%m-%d"),
        })
        self.flare_events = events
        print(f"  [DONKI] {len(events)} flare(s) encontrados.")
        return events

    def fetch_donki_geostorm(self) -> list:
        """Tempestades geomagnéticas nos últimos N dias."""
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=DATA_WINDOW_DAYS)
        print("  [DONKI] Buscando Tempestades Geomagnéticas...")
        events = self._donki_get("GST", {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate":   end.strftime("%Y-%m-%d"),
        })
        self.storm_events = events
        print(f"  [DONKI] {len(events)} tempestade(s) encontradas.")
        return events

    def fetch_donki_enlil(self) -> list:
        """Simulações WSA-Enlil do fluxo de vento solar até a Lua."""
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=DATA_WINDOW_DAYS)
        print("  [DONKI] Buscando simulações WSA-Enlil...")
        sims = self._donki_get("WSAEnlilSimulations", {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate":   end.strftime("%Y-%m-%d"),
        })
        print(f"  [DONKI] {len(sims)} simulação(ões) Enlil.")
        return sims

    # ── Análise ────────────────────────────────────────────────────

    def calculate_lunar_flux(self) -> dict:
        """
        Estima o fluxo de vento solar na superfície lunar.
        A Lua não tem campo magnético global → vento solar atinge diretamente.
        """
        if self.plasma_data is None or self.plasma_data.empty:
            return {}

        df = self.plasma_data.dropna(subset=["speed", "density"])
        if df.empty:
            return {}

        # Fluxo de partículas: F = n × v  (partículas / cm² / s)
        df = df.copy()
        df["flux"] = df["density"] * df["speed"] * 1e5  # conversão km/s → cm/s

        atual   = df.iloc[-1]
        media   = df.mean(numeric_only=True)
        maximo  = df.max(numeric_only=True)

        # Classificação da intensidade
        v = float(atual.get("speed", 0))
        if   v > 700: nivel = "EXTREMO"
        elif v > 550: nivel = "ALTO"
        elif v > 400: nivel = "MODERADO"
        else:          nivel = "BAIXO"

        return {
            "velocidade_atual_kms": round(v, 1),
            "densidade_atual_pcm3": round(float(atual.get("density", 0)), 2),
            "temperatura_atual_K":  round(float(atual.get("temperature", 0)), 0),
            "fluxo_atual":          round(float(df["flux"].iloc[-1]), 2),
            "velocidade_media_kms": round(float(media.get("speed", 0)), 1),
            "velocidade_max_kms":   round(float(maximo.get("speed", 0)), 1),
            "nivel_atividade":      nivel,
            "total_cmes":           len(self.cme_events),
            "total_flares":         len(self.flare_events),
            "total_tempestades":    len(self.storm_events),
        }

    def fetch_all(self) -> dict:
        """Busca todos os dados solares disponíveis."""
        print("\n━━━ MÓDULO 1: VENTO SOLAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.fetch_noaa_plasma()
        self.fetch_noaa_magnetic()
        self.fetch_donki_cme()
        self.fetch_donki_flares()
        self.fetch_donki_geostorm()
        return self.calculate_lunar_flux()


# ══════════════════════════════════════════════════════════════════
#  MÓDULO 2 — MINERALOGIA LUNAR (CMR / PDS / USGS)
# ══════════════════════════════════════════════════════════════════
class LunarMineralogyData:
    """
    Acessa dados de ilmenita / TiO₂ da Lua via:
    • NASA CMR (M3 Chandrayaan-1)
    • NASA PDS (LRO, Lunar Prospector)
    • USGS Astrogeology
    """

    def __init__(self):
        self.m3_granules   = []
        self.pds_datasets  = []
        self.ilmenite_map  = None  # grade numpy lon×lat

    # ── NASA CMR ───────────────────────────────────────────────────

    def search_cmr_m3(self) -> list:
        """
        Busca granules do Moon Mineralogy Mapper (M3) no CMR.
        M3 detecta ilmenita via espectroscopia 0.4–3.0 µm.
        """
        print("  [CMR] Buscando granules M3 (Chandrayaan-1)...")
        params = {
            "short_name":  "M3T20110921_RFL",   # Reflectância calibrada M3
            "page_size":   10,
            "page_num":    1,
        }
        headers = {}
        if EARTHDATA_TOKEN:
            headers["Authorization"] = f"Bearer {EARTHDATA_TOKEN}"
        try:
            resp = requests.get(
                f"{CMR_BASE}/granules.json",
                params=params,
                headers=headers,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            granules = data.get("feed", {}).get("entry", [])
            self.m3_granules = granules
            print(f"  [CMR] {len(granules)} granule(s) M3 encontrados.")
            return granules
        except Exception as e:
            print(f"  [CMR] Erro M3: {e}")
            return []

    def search_cmr_lunar_prospector(self) -> list:
        """
        Busca dados do Lunar Prospector GRS (Gamma-Ray Spectrometer).
        Mede abundância de Fe, Ti, Th, K — diretamente ligados ao He-3 implantado.
        """
        print("  [CMR] Buscando Lunar Prospector GRS...")
        params = {
            "short_name": "LP_GRS",
            "page_size":  5,
        }
        headers = {}
        if EARTHDATA_TOKEN:
            headers["Authorization"] = f"Bearer {EARTHDATA_TOKEN}"
        try:
            resp = requests.get(
                f"{CMR_BASE}/granules.json",
                params=params,
                headers=headers,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            print(f"  [CMR] {len(entries)} dataset(s) Lunar Prospector.")
            return entries
        except Exception as e:
            print(f"  [CMR] Erro Lunar Prospector: {e}")
            return []

    def search_pds_datasets(self) -> list:
        """
        Busca datasets lunares no PDS (Planetary Data System) da NASA.
        """
        print("  [PDS] Buscando datasets de TiO₂ / ilmenita...")
        params = {
            "q":          "titanium moon ilmenite",
            "limit":      5,
            "fields":     "lidvid,title,description",
        }
        try:
            resp = requests.get(
                f"{PDS_BASE}/products",
                params=params,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            self.pds_datasets = items
            print(f"  [PDS] {len(items)} produto(s) encontrado(s).")
            return items
        except Exception as e:
            print(f"  [PDS] Erro: {e}")
            return []

    def search_jaxa_selene(self) -> dict:
        """
        Consulta metadados do JAXA SELENE/Kaguya via DARTS.
        GRS mediu Ti e Fe com resolução de 0.5°/pixel.
        """
        print("  [JAXA] Consultando SELENE/Kaguya DARTS...")
        url = "https://darts.isas.jaxa.jp/planet/pdap/selene/api/dataset_list.json"
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            print(f"  [JAXA] Catálogo SELENE disponível: {len(data.get('datasets', []))} datasets.")
            return data
        except Exception as e:
            print(f"  [JAXA] Aviso ({e}) — usando dados de referência compilados.")
            return {}

    # ── Modelagem ─────────────────────────────────────────────────

    def estimate_he3_from_tio2(self, tio2_pct: float, solar_flux: float = 1.0) -> float:
        """
        Estima concentração de He-3 (ppb) a partir de TiO₂ (% peso).

        Modelo baseado em:
        • Haskin & Warren (1991): correlação TiO₂ × retenção He-3
        • Fa & Jin (2007): implantação de He-3 pelo vento solar
        • Fator de fluxo solar normalizado (1.0 = condição média)

        Fórmula simplificada:
            [He-3] ≈ (2.4 × TiO₂ + 0.5) × fluxo_solar  (ppb)
        """
        base_he3 = 2.4 * tio2_pct + 0.5
        return round(base_he3 * solar_flux, 2)

    def generate_ilmenite_map(self, solar_flux: float = 1.0) -> np.ndarray:
        """
        Gera mapa global de Ilmenita/He-3 da Lua (grade 360×180°).
        Baseado nas regiões catalogadas de SELENE/Clementine/M3.
        """
        print("  [MAPA] Gerando mapa de ilmenita/He-3 lunar...")

        # Grade lon [-180, 180), lat [-90, 90)  → 1°/pixel
        lons = np.linspace(-180, 180, 360)
        lats = np.linspace(-90,   90, 180)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        tio2_map = np.zeros_like(lon_grid)

        # Plumas gaussianas para cada região catalogada
        for reg in LUNAR_REGIONS:
            dist2 = ((lon_grid - reg["lon"]) ** 2 + (lat_grid - reg["lat"]) ** 2)
            sigma = reg["raio"] / 2.5
            tio2_map += reg["tio2"] * np.exp(-dist2 / (2 * sigma ** 2))

        # Clamp: TiO₂ máximo realista ~13%
        tio2_map = np.clip(tio2_map, 0, 13)

        # Converte para He-3 (ppb)
        he3_map = 2.4 * tio2_map + 0.5 * (tio2_map > 0.5)
        he3_map *= solar_flux
        he3_map = np.clip(he3_map, 0, 40)

        self.ilmenite_map = he3_map
        print("  [MAPA] Mapa de He-3 gerado (360×180 pixels).")
        return he3_map

    def get_top_regions(self, solar_flux: float = 1.0) -> pd.DataFrame:
        """Retorna tabela das regiões com maior potencial de He-3."""
        rows = []
        for r in LUNAR_REGIONS:
            he3 = self.estimate_he3_from_tio2(r["tio2"], solar_flux)
            rows.append({
                "Região":        r["nome"],
                "Longitude (°)": r["lon"],
                "Latitude (°)":  r["lat"],
                "TiO₂ (% peso)": r["tio2"],
                "He-3 est. (ppb)": he3,
                "Classficação":  "🔴 ALTO" if he3 > 20 else ("🟡 MÉDIO" if he3 > 10 else "🟢 BAIXO"),
            })
        df = pd.DataFrame(rows).sort_values("He-3 est. (ppb)", ascending=False)
        return df

    def fetch_all(self, solar_flux: float = 1.0) -> dict:
        """Busca todos os dados lunares."""
        print("\n━━━ MÓDULO 2: MINERALOGIA LUNAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.search_cmr_m3()
        self.search_cmr_lunar_prospector()
        self.search_pds_datasets()
        self.search_jaxa_selene()
        he3_map = self.generate_ilmenite_map(solar_flux)
        return {"he3_map": he3_map, "regions": self.get_top_regions(solar_flux)}


# ══════════════════════════════════════════════════════════════════
#  MÓDULO 3 — VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════
class Dashboard:
    """Cria o painel visual do monitor."""

    def __init__(self, solar: SolarWindMonitor, lunar: LunarMineralogyData):
        self.solar = solar
        self.lunar = lunar

    # ── Paletas ───────────────────────────────────────────────────

    @staticmethod
    def _he3_cmap():
        """Gradiente visual para o mapa de He-3."""
        cores = [
            (0.04, 0.04, 0.12),   # quase preto (zero)
            (0.00, 0.20, 0.50),   # azul escuro
            (0.00, 0.55, 0.80),   # azul claro
            (0.00, 0.80, 0.60),   # verde-água
            (0.90, 0.80, 0.00),   # amarelo
            (1.00, 0.40, 0.00),   # laranja
            (1.00, 0.10, 0.10),   # vermelho
        ]
        return LinearSegmentedColormap.from_list("he3", cores, N=256)

    # ── Painel Solar ──────────────────────────────────────────────

    def plot_solar(self, ax_speed, ax_mag, ax_info):
        """Gráficos de vento solar."""
        df_p = self.solar.plasma_data
        df_m = self.solar.mag_data

        # ─ Velocidade ─
        if df_p is not None and not df_p.empty:
            ax_speed.plot(df_p["time_tag"], df_p["speed"],
                          color=COR_AZUL, lw=1.2, label="Velocidade (km/s)")
            ax_speed.fill_between(df_p["time_tag"], df_p["speed"],
                                  alpha=0.15, color=COR_AZUL)
            # Linhas de referência
            for v, lbl, cor in [(400, "Lento", COR_VERDE), (600, "Rápido", COR_LARANJA)]:
                ax_speed.axhline(v, ls="--", lw=0.8, color=cor, alpha=0.6, label=lbl)
            ax_speed.set_title("Velocidade do Vento Solar (NOAA DSCOVR)",
                               color=COR_TEXTO, fontsize=10)
            ax_speed.set_ylabel("km/s", color=COR_TEXTO)
            ax_speed.tick_params(colors=COR_TEXTO)
            ax_speed.set_facecolor(COR_FUNDO)
            ax_speed.spines[:].set_color(COR_CINZA)
            ax_speed.legend(fontsize=7, framealpha=0.2, labelcolor=COR_TEXTO)
            ax_speed.yaxis.label.set_color(COR_TEXTO)
        else:
            ax_speed.text(0.5, 0.5, "Dados indisponíveis", ha="center",
                          va="center", color=COR_TEXTO, transform=ax_speed.transAxes)
            ax_speed.set_facecolor(COR_FUNDO)

        # ─ Campo magnético Bz ─
        if df_m is not None and not df_m.empty and "bz_gsm" in df_m.columns:
            bz = df_m["bz_gsm"]
            cols = [COR_LARANJA if v < 0 else COR_AZUL for v in bz]
            ax_mag.bar(df_m["time_tag"], bz, color=cols,
                       width=0.002, alpha=0.8, label="Bz IMF (nT)")
            ax_mag.axhline(0, color=COR_CINZA, lw=0.7)
            ax_mag.set_title("Campo Magnético Interplanetário Bz",
                             color=COR_TEXTO, fontsize=10)
            ax_mag.set_ylabel("nT", color=COR_TEXTO)
            ax_mag.tick_params(colors=COR_TEXTO)
            ax_mag.set_facecolor(COR_FUNDO)
            ax_mag.spines[:].set_color(COR_CINZA)
            # Legenda manual
            p1 = mpatches.Patch(color=COR_LARANJA, label="Sul (desfavorável)")
            p2 = mpatches.Patch(color=COR_AZUL,    label="Norte (favorável)")
            ax_mag.legend(handles=[p1, p2], fontsize=7,
                          framealpha=0.2, labelcolor=COR_TEXTO)
        else:
            ax_mag.text(0.5, 0.5, "Dados IMF indisponíveis", ha="center",
                        va="center", color=COR_TEXTO, transform=ax_mag.transAxes)
            ax_mag.set_facecolor(COR_FUNDO)

        # ─ Painel de informações ─
        flux = self.solar.calculate_lunar_flux()
        ax_info.set_facecolor(COR_FUNDO)
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
        ax_info.axis("off")

        nivel = flux.get("nivel_atividade", "N/D")
        cor_nivel = {
            "EXTREMO": "#ff2222", "ALTO": COR_LARANJA,
            "MODERADO": COR_AMARELO, "BAIXO": COR_VERDE,
        }.get(nivel, COR_TEXTO)

        linhas = [
            ("━━━ STATUS DO VENTO SOLAR ━━━", COR_TEXTO, 11, "bold"),
            (f"Velocidade atual: {flux.get('velocidade_atual_kms','—')} km/s", COR_AZUL, 10, "normal"),
            (f"Densidade:        {flux.get('densidade_atual_pcm3','—')} p/cm³", COR_AZUL, 10, "normal"),
            (f"Temperatura:      {flux.get('temperatura_atual_K','—')} K",   COR_AZUL, 10, "normal"),
            (f"Velocidade média: {flux.get('velocidade_media_kms','—')} km/s", COR_TEXTO, 9, "normal"),
            (f"Velocidade máx:   {flux.get('velocidade_max_kms','—')} km/s",  COR_TEXTO, 9, "normal"),
            ("", "", 9, "normal"),
            (f"Nível de atividade: {nivel}", cor_nivel, 11, "bold"),
            ("", "", 9, "normal"),
            ("━━━ EVENTOS ESPAÇO-CLIMÁTICOS ━━━", COR_TEXTO, 11, "bold"),
            (f"CMEs detectadas:      {flux.get('total_cmes','—')}",       COR_LARANJA, 10, "normal"),
            (f"Flares solares:       {flux.get('total_flares','—')}",     COR_AMARELO, 10, "normal"),
            (f"Tempestades geomag:   {flux.get('total_tempestades','—')}", COR_ROXO,   10, "normal"),
            ("", "", 9, "normal"),
            (f"Janela analisada: {DATA_WINDOW_DAYS} dias", COR_CINZA, 8, "normal", "italic"),
            (f"Atualizado: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", COR_CINZA, 8, "normal", "italic"),
        ]

        # garante 5 elementos em cada tupla (txt, cor, fontsize, fontweight, fontstyle)
        linhas = [t if len(t) == 5 else t + ("normal",) for t in linhas]

        y = 0.97
        for txt, cor, fs, fw, fst in linhas:
            if txt == "":
                y -= 0.055
                continue
            ax_info.text(0.05, y, txt, color=cor, fontsize=fs,
                         fontweight=fw, fontstyle=fst,
                         transform=ax_info.transAxes, va="top")
            y -= 0.065

    # ── Mapa Lunar ────────────────────────────────────────────────

    def plot_lunar_map(self, ax_map, ax_table):
        """Mapa de He-3 potencial na superfície lunar."""
        he3_map = self.lunar.ilmenite_map
        cmap    = self._he3_cmap()

        if he3_map is not None:
            im = ax_map.imshow(
                he3_map,
                extent=[-180, 180, -90, 90],
                origin="lower",
                cmap=cmap,
                vmin=0, vmax=32,
                aspect="auto",
            )
            # Marcadores das regiões
            for reg in LUNAR_REGIONS:
                cor = "#ff4444" if reg["he3_ppb"] > 20 else (
                      "#ffaa00" if reg["he3_ppb"] > 10 else "#44ff88")
                ax_map.scatter(reg["lon"], reg["lat"], s=60,
                               color=cor, edgecolors="white",
                               linewidths=0.5, zorder=5)
                ax_map.annotate(
                    reg["nome"].replace("Mare ", "M. "),
                    xy=(reg["lon"], reg["lat"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=5.5, color="white", alpha=0.9,
                )

            # Grade
            ax_map.set_xticks(range(-180, 181, 30))
            ax_map.set_yticks(range(-90, 91, 30))
            ax_map.tick_params(colors=COR_TEXTO, labelsize=7)
            ax_map.set_xlabel("Longitude (°)", color=COR_TEXTO, fontsize=8)
            ax_map.set_ylabel("Latitude (°)",  color=COR_TEXTO, fontsize=8)
            ax_map.set_title(
                "Mapa de He-3 Potencial — Lua (ppb)\n"
                "Baseado em TiO₂/Ilmenita: M3 (Chandrayaan-1) · SELENE/Kaguya · Lunar Prospector",
                color=COR_TEXTO, fontsize=9,
            )
            ax_map.set_facecolor(COR_FUNDO)
            ax_map.spines[:].set_color(COR_CINZA)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax_map, orientation="horizontal",
                                 fraction=0.03, pad=0.12)
            cbar.set_label("He-3 Estimado (ppb)", color=COR_TEXTO, fontsize=8)
            cbar.ax.tick_params(colors=COR_TEXTO, labelsize=7)
        else:
            ax_map.text(0.5, 0.5, "Mapa não gerado", ha="center",
                        va="center", color=COR_TEXTO)
            ax_map.set_facecolor(COR_FUNDO)

        # ─ Tabela de regiões ─
        flux  = self.solar.calculate_lunar_flux()
        sfact = flux.get("velocidade_atual_kms", 450) / 450
        df    = self.lunar.get_top_regions(solar_flux=sfact)

        ax_table.set_facecolor(COR_FUNDO)
        ax_table.axis("off")

        cols = ["Região", "TiO₂ (%)", "He-3 (ppb)", "Clasf."]
        rows = []
        for _, row in df.iterrows():
            rows.append([
                row["Região"],
                f"{row['TiO₂ (% peso)']:.1f}",
                f"{row['He-3 est. (ppb)']:.1f}",
                row["Classficação"],
            ])

        tab = ax_table.table(
            cellText=rows,
            colLabels=cols,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)

        for (r, c), cell in tab.get_celld().items():
            cell.set_facecolor("#0e1030" if r % 2 == 0 else "#080820")
            cell.set_edgecolor(COR_CINZA)
            if r == 0:
                cell.set_facecolor("#1a1a50")
                cell.set_text_props(color=COR_AMARELO, fontweight="bold")
            else:
                cell.set_text_props(color=COR_TEXTO)

    # ── Dashboard completo ────────────────────────────────────────

    def render(self, save: bool = True) -> Path:
        """Gera o painel completo e salva como PNG."""
        print("\n━━━ MÓDULO 3: RENDERIZANDO DASHBOARD ━━━━━━━━━━━━━━━━━━━━━━")

        fig = plt.figure(figsize=(20, 14), facecolor=COR_FUNDO)
        fig.suptitle(
            "🌑  LUNAR HE-3 & SOLAR WIND MONITOR  ☀️\n"
            f"Dados de {DATA_WINDOW_DAYS} dias  ·  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            color=COR_AMARELO, fontsize=14, fontweight="bold", y=0.98,
        )

        gs = gridspec.GridSpec(
            3, 3,
            figure=fig,
            left=0.06, right=0.97,
            top=0.93,  bottom=0.06,
            hspace=0.45, wspace=0.30,
        )

        ax_speed = fig.add_subplot(gs[0, :2])
        ax_mag   = fig.add_subplot(gs[1, :2])
        ax_info  = fig.add_subplot(gs[0:2, 2])
        ax_map   = fig.add_subplot(gs[2, :2])
        ax_table = fig.add_subplot(gs[2, 2])

        for ax in [ax_speed, ax_mag, ax_info, ax_map, ax_table]:
            ax.set_facecolor(COR_FUNDO)

        self.plot_solar(ax_speed, ax_mag, ax_info)
        self.plot_lunar_map(ax_map, ax_table)

        path = OUTPUT_DIR / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        if save:
            fig.savefig(path, dpi=120, facecolor=COR_FUNDO, bbox_inches="tight")
            print(f"  [OK] Dashboard salvo → {path}")
        plt.show()
        plt.close(fig)
        return path


# ══════════════════════════════════════════════════════════════════
#  MÓDULO 4 — EXPORTAÇÃO DE DADOS
# ══════════════════════════════════════════════════════════════════
class DataExporter:
    """Exporta todos os dados coletados em CSV."""

    def __init__(self, solar: SolarWindMonitor, lunar: LunarMineralogyData):
        self.solar = solar
        self.lunar = lunar

    def export_solar_wind(self) -> Path:
        if self.solar.plasma_data is not None:
            p = OUTPUT_DIR / "solar_wind_plasma.csv"
            self.solar.plasma_data.to_csv(p, index=False)
            print(f"  [EXPORT] {p}")
        if self.solar.mag_data is not None:
            p = OUTPUT_DIR / "solar_wind_mag.csv"
            self.solar.mag_data.to_csv(p, index=False)
            print(f"  [EXPORT] {p}")

    def export_cme_events(self) -> Path:
        if self.solar.cme_events:
            p = OUTPUT_DIR / "cme_events.json"
            with open(p, "w") as f:
                json.dump(self.solar.cme_events, f, indent=2, default=str)
            print(f"  [EXPORT] {p}")

    def export_flare_events(self) -> Path:
        if self.solar.flare_events:
            p = OUTPUT_DIR / "flare_events.json"
            with open(p, "w") as f:
                json.dump(self.solar.flare_events, f, indent=2, default=str)
            print(f"  [EXPORT] {p}")

    def export_lunar_regions(self) -> Path:
        flux  = self.solar.calculate_lunar_flux()
        sfact = flux.get("velocidade_atual_kms", 450) / 450
        df    = self.lunar.get_top_regions(solar_flux=sfact)
        p     = OUTPUT_DIR / "lunar_ilmenite_regions.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"  [EXPORT] {p}")
        return p

    def export_he3_map_csv(self) -> Path:
        """Exporta o mapa de He-3 como CSV (lon, lat, he3_ppb)."""
        if self.lunar.ilmenite_map is None:
            return None
        lons = np.linspace(-180, 180, 360)
        lats = np.linspace(-90, 90, 180)
        rows = []
        step = 5  # amostragem a cada 5° para arquivo menor
        for i in range(0, 180, step):
            for j in range(0, 360, step):
                rows.append({
                    "longitude": round(lons[j], 1),
                    "latitude":  round(lats[i], 1),
                    "he3_ppb":   round(float(self.lunar.ilmenite_map[i, j]), 3),
                })
        df = pd.DataFrame(rows)
        p  = OUTPUT_DIR / "he3_map_sample.csv"
        df.to_csv(p, index=False)
        print(f"  [EXPORT] {p} ({len(df)} amostras)")
        return p

    def export_m3_granules(self) -> Path:
        if self.lunar.m3_granules:
            p = OUTPUT_DIR / "m3_granules.json"
            with open(p, "w") as f:
                json.dump(self.lunar.m3_granules, f, indent=2, default=str)
            print(f"  [EXPORT] {p}")

    def export_all(self):
        print("\n━━━ MÓDULO 4: EXPORTANDO DADOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.export_solar_wind()
        self.export_cme_events()
        self.export_flare_events()
        self.export_lunar_regions()
        self.export_he3_map_csv()
        self.export_m3_granules()
        print(f"  [OK] Todos os arquivos salvos em: {OUTPUT_DIR.resolve()}")


# ══════════════════════════════════════════════════════════════════
#  PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Lunar He-3 & Solar Wind Monitor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--solar",  action="store_true", help="Apenas dados de vento solar")
    p.add_argument("--lunar",  action="store_true", help="Apenas dados lunares")
    p.add_argument("--export", action="store_true", help="Exportar CSVs/JSONs")
    p.add_argument("--no-plot", action="store_true", help="Não exibir gráfico")
    return p.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   LUNAR HE-3 & SOLAR WIND MONITOR  — Iniciando...   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Saída: {OUTPUT_DIR.resolve()}")
    print(f"  NASA API: {'✓ configurada' if NASA_API_KEY != 'DEMO_KEY' else '⚠ usando DEMO_KEY'}")
    print(f"  Earthdata: {'✓ token presente' if EARTHDATA_TOKEN else '⚠ token ausente'}")

    solar = SolarWindMonitor()
    lunar = LunarMineralogyData()

    if args.solar or (not args.lunar):
        flux = solar.fetch_all()

    sfact = flux.get("velocidade_atual_kms", 450) / 450 if "flux" in dir() or True else 1.0
    if "flux" in dir():
        sfact = flux.get("velocidade_atual_kms", 450) / 450

    if args.lunar or (not args.solar):
        lunar.fetch_all(solar_flux=sfact)

    if args.export:
        exporter = DataExporter(solar, lunar)
        exporter.export_all()

    if not args.no_plot:
        dash = Dashboard(solar, lunar)
        dash.render(save=True)

    # ── Resumo final ─
    print("\n━━━ RESUMO FINAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    flux_info = solar.calculate_lunar_flux()
    if flux_info:
        print(f"  🌬️  Vento solar: {flux_info.get('velocidade_atual_kms')} km/s "
              f"[{flux_info.get('nivel_atividade')}]")
        print(f"  ☀️  CMEs: {flux_info.get('total_cmes')}  |  "
              f"Flares: {flux_info.get('total_flares')}  |  "
              f"Tempestades: {flux_info.get('total_tempestades')}")

    print("\n  🌑 Top 3 regiões com maior potencial de He-3:")
    df = lunar.get_top_regions(solar_flux=sfact)
    for _, row in df.head(3).iterrows():
        print(f"     • {row['Região']:<26}  "
              f"TiO₂={row['TiO₂ (% peso)']:.1f}%  "
              f"He-3≈{row['He-3 est. (ppb)']:.1f} ppb  "
              f"{row['Classficação']}")

    print("\n  ✅ Concluído. Gráficos salvos em:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()