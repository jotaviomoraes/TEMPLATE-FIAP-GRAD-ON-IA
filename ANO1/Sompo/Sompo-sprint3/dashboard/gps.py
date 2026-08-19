"""
gps.py
------
Captura localizacao e velocidade em tempo real a partir do GPS do celular
do operador, usando a Geolocation API do proprio navegador (funciona no
Chrome/Safari mobile quando o operador abre o dashboard no celular).

Requer: pip install streamlit-js-eval

Duas fontes de velocidade:
1) `coords.speed` -- vem direto do chip GPS (m/s), quando o navegador/
   aparelho fornece. E o mais preciso, mas pode vir None em alguns
   dispositivos ou quando o aparelho esta parado.
2) Fallback: calculo manual via formula de Haversine entre a leitura
   atual e a leitura anterior guardada em st.session_state, dividido
   pelo intervalo de tempo.
"""

from __future__ import annotations

import time
from math import radians, sin, cos, sqrt, atan2

import streamlit as st
from streamlit_js_eval import get_geolocation


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distancia em km entre dois pontos GPS."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def capturar_posicao(ciclo: int | str = 0) -> dict | None:
    """
    Dispara o prompt de permissao de localizacao no navegador (na primeira
    vez) e retorna a leitura mais recente.

    IMPORTANTE: `get_geolocation()` do streamlit-js-eval usa por padrao uma
    key FIXA ("getLocation()"). Como componentes do Streamlit so reexecutam
    o JS quando a key muda, sem variar `ciclo` a cada rerun a leitura fica
    travada na primeira posicao para sempre -- e por isso a velocidade
    calculada nunca avancava (sempre so 1 ponto disponivel).

    Passe aqui o contador do st_autorefresh (ou qualquer valor que mude a
    cada rerun) para forcar uma nova leitura de GPS em cada ciclo.

    Retorna None se o navegador ainda nao respondeu / usuario negou.
    """
    loc = get_geolocation(component_key=f"geo_{ciclo}")
    if not loc or "coords" not in loc:
        return None

    coords = loc["coords"]
    return {
        "lat": coords.get("latitude"),
        "lon": coords.get("longitude"),
        "accuracy_m": coords.get("accuracy"),
        "speed_ms": coords.get("speed"),  # pode ser None
        "timestamp": loc.get("timestamp", time.time() * 1000) / 1000,
    }


def obter_velocidade_kmh(posicao_atual: dict) -> tuple[float | None, str]:
    """
    Retorna (velocidade_kmh, origem) onde origem in {"gps_chip", "calculada", "indisponivel"}.
    Usa e atualiza st.session_state["_ultima_posicao"] para o fallback.
    """
    if posicao_atual is None:
        return None, "indisponivel"

    # 1) Velocidade direta do chip GPS, se disponivel
    if posicao_atual.get("speed_ms") is not None:
        return round(posicao_atual["speed_ms"] * 3.6, 1), "gps_chip"

    # 2) Fallback: calcular a partir da leitura anterior
    anterior = st.session_state.get("_ultima_posicao")
    st.session_state["_ultima_posicao"] = posicao_atual

    if anterior is None:
        return None, "indisponivel"  # ainda nao ha 2 pontos para comparar

    dt_h = (posicao_atual["timestamp"] - anterior["timestamp"]) / 3600
    if dt_h <= 0:
        return None, "indisponivel"

    dist_km = _haversine_km(
        anterior["lat"], anterior["lon"],
        posicao_atual["lat"], posicao_atual["lon"],
    )
    velocidade = dist_km / dt_h
    # Filtro de ruido: GPS parado costuma "derivar" alguns km/h por erro de precisao
    if velocidade < 0.5:
        velocidade = 0.0
    return round(velocidade, 1), "calculada"
