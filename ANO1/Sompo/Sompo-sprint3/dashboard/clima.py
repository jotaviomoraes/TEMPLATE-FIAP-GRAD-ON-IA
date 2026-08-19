"""
clima.py
--------
Integracao com a API climatica Open-Meteo (gratuita, sem API key).
Substitui os valores simulados de `temperatura` e `chuva_24h` por dados
reais, a partir da latitude/longitude capturada pelo GPS do operador.

Documentacao da API: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import requests
import streamlit as st

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=600, show_spinner=False)  # cache de 10 min por coordenada
def buscar_clima_atual(lat: float, lon: float) -> dict:
    """
    Consulta a Open-Meteo para a posicao atual e retorna um dicionario
    compativel com as features do modelo:

        {
            "temperatura": float (°C),
            "chuva_24h":   float (mm, soma das ultimas 24h),
            "umidade_solo": float (%, aproximada via camada 0-1cm) | None,
            "fonte": "open-meteo",
            "ok": bool,
        }

    Se a chamada falhar, retorna ok=False e valores None -- quem chama
    decide o fallback (ex.: manter ultimo valor valido ou pedir input manual).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation",
        "hourly": "precipitation,soil_moisture_0_to_1cm",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        temperatura = data["current"]["temperature_2m"]

        # Soma das precipitacoes horarias das ultimas 24h ate agora
        horarios = data["hourly"]["time"]
        chuva_hora = data["hourly"]["precipitation"]
        agora = data["current"]["time"]
        idx_agora = horarios.index(agora) if agora in horarios else len(horarios) - 1
        janela = chuva_hora[max(0, idx_agora - 23): idx_agora + 1]
        chuva_24h = round(sum(v for v in janela if v is not None), 1)

        # Umidade do solo (0-1cm) -- aproximacao; fracao volumetrica 0-1 -> %
        umidade_solo = None
        if "soil_moisture_0_to_1cm" in data["hourly"]:
            solo_serie = data["hourly"]["soil_moisture_0_to_1cm"]
            valor_solo = solo_serie[idx_agora] if idx_agora < len(solo_serie) else None
            if valor_solo is not None:
                umidade_solo = round(valor_solo * 100, 1)

        return {
            "temperatura": temperatura,
            "chuva_24h": chuva_24h,
            "umidade_solo": umidade_solo,
            "fonte": "open-meteo",
            "ok": True,
        }

    except (requests.RequestException, KeyError, ValueError) as e:
        return {
            "temperatura": None,
            "chuva_24h": None,
            "umidade_solo": None,
            "fonte": "open-meteo",
            "ok": False,
            "erro": str(e),
        }
