"""
app.py - Dashboard Sompo AgroPredict
-------------------------------------
Dashboard funcional em Streamlit que consome o modelo Random Forest
treinado (modelo.pkl) e exibe predicao em tempo real, feature importance
e historico de predicoes.

Limiares de ALERTA por variavel:
  umidade_solo > 30, declividade > 18, chuva_24h > 10,
  temperatura > 38, velocidade > 10, idade_maquina > 10, horas_operacao > 8

Limiares MAXIMO (override imediato para ALTO, independente do modelo):
  umidade_solo >= 45, declividade >= 25, chuva_24h >= 40,
  temperatura >= 45, velocidade >= 20

Modo de execucao:
    streamlit run app.py
"""
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from clima import buscar_clima_atual
from gps import capturar_posicao, obter_velocidade_kmh
from persistencia import registrar_local, registrar_oracle, carregar_local, contar_local

# =====================================================================
# Configuracao geral
# =====================================================================
st.set_page_config(
    page_title="Sompo AgroPredict",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR    = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "modelo" / "modelo.pkl"

CONFIG_ORACLE = {
    "user":     os.getenv("ORACLE_USER",    "system"),
    "password": os.getenv("ORACLE_PASSWORD","oracle"),
    "dsn":      os.getenv("ORACLE_DSN",     "localhost:1521/XE"),
}

CORES_RISCO = {"BAIXO": "#2C5F2D", "MEDIO": "#E8A33D", "ALTO": "#C8443D"}
EMOJI_RISCO = {"BAIXO": "🟢",      "MEDIO": "🟡",      "ALTO": "🔴"}

# Limiares de alerta e maximo por variavel
ALERTA = {
    "umidade_solo":   30,
    "declividade":    18,
    "chuva_24h":      10,
    "temperatura":    38,
    "velocidade":     10,
    "idade_maquina":  10,
    "horas_operacao":  8,
}
MAXIMO = {
    "umidade_solo":  45,
    "declividade":   25,
    "chuva_24h":     40,
    "temperatura":   45,
    "velocidade":    20,
}


# =====================================================================
# Helpers
# =====================================================================
def verificar_max(row: dict) -> tuple[bool, list[str]]:
    """Retorna (em_max, lista de variaveis no MAX)."""
    vars_max = [v for v, lim in MAXIMO.items() if row.get(v, 0) >= lim]
    return len(vars_max) > 0, vars_max


def contar_alertas(row: dict) -> tuple[int, list[str]]:
    """Retorna (n_alertas, lista de variaveis em alerta)."""
    vars_alerta = [v for v, lim in ALERTA.items() if row.get(v, 0) > lim]
    return len(vars_alerta), vars_alerta


def gerar_recomendacao(row: dict, classe: str, vars_max: list) -> str:
    if classe == "BAIXO":
        return "Seguir rota atual"
    if "umidade_solo" in vars_max or (row["umidade_solo"] > 70 and row["chuva_24h"] > 20):
        return "Risco de atolamento — evitar areas saturadas"
    if "declividade" in vars_max or (row["velocidade"] > 15 and row["declividade"] > 18):
        return "Atencao a inclinacao — reduzir velocidade imediatamente"
    if "chuva_24h" in vars_max:
        return "Chuva intensa — suspender operacao"
    if "temperatura" in vars_max or (row["temperatura"] > 38 and row["horas_operacao"] > 8):
        return "Pausa obrigatoria — risco de fadiga e superaquecimento"
    if "velocidade" in vars_max:
        return "Velocidade critica — reduzir imediatamente"
    if row["idade_maquina"] > 10 and row["horas_operacao"] > 8:
        return "Verificar equipamento — desgaste elevado"
    if classe == "ALTO":
        return "Interromper operacao"
    return "Reduzir velocidade e monitorar"


def label_alerta(valor, chave):
    """Retorna delta string para st.metric com icone coerente."""
    if chave in MAXIMO and valor >= MAXIMO[chave]:
        return "🔴 MAX"
    if valor > ALERTA[chave]:
        return "⚠️ Alerta"
    return "✅ OK"


def formatar_hhmm(horas: float) -> str:
    """Converte horas decimais (ex.: 1.5) para 'HH:MM' (ex.: '01:30')."""
    total_min = int(round(max(horas, 0) * 60))
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"


# =====================================================================
# Cache de recursos
# =====================================================================
@st.cache_resource
def carregar_modelo():
    if not MODELO_PATH.exists():
        return None
    return joblib.load(MODELO_PATH)


@st.cache_data
def buscar_historico_oracle():
    try:
        import oracledb
        conn = oracledb.connect(**CONFIG_ORACLE)
        df = pd.read_sql("""
            SELECT id_predicao, data_hora, classe_risco, score_risco,
                   umidade_solo, declividade, chuva_24h, velocidade,
                   status_operacao, recomendacao
            FROM predicoes
            ORDER BY data_hora DESC
            FETCH FIRST 100 ROWS ONLY
        """, conn)
        conn.close()
        return df
    except Exception:
        return None


# =====================================================================
# Sidebar — Inputs do operador
# =====================================================================
st.sidebar.title("🌾 Sompo AgroPredict")
st.sidebar.markdown("**Calculadora de risco operacional**")
st.sidebar.markdown("---")

modo_real = st.sidebar.toggle(
    "📡 Usar GPS + clima em tempo real", value=True,
    help="Desative para digitar tudo manualmente (modo demo/offline)."
)

ciclo = st_autorefresh(interval=10_000, key="_refresh_telemetria") if modo_real else 0
# ↑ 10s. O `ciclo` (contador do autorefresh) é passado pro GPS abaixo pra
# forçar uma NOVA leitura a cada rerun — sem isso, get_geolocation() usa
# sempre a mesma key e trava na 1ª posição, e a velocidade nunca é calculada.

# ---- Toggles de override manual, campo a campo ----------------------
# Cobre o caso do operador ter esquecido de ligar o GPS/rede: cada campo
# pode ser assumido manualmente sem precisar desligar tudo.
with st.sidebar.expander("✍️ Substituir algum campo manualmente"):
    st.caption("Marque se o sensor correspondente não conectou hoje.")
    manual_velocidade = st.checkbox("Velocidade", key="manual_velocidade")
    manual_clima      = st.checkbox("Temperatura / Chuva", key="manual_clima")
    manual_umidade    = st.checkbox("Umidade do solo", key="manual_umidade")
    manual_horas      = st.checkbox("Horas de operação", key="manual_horas")

# ---- Captura de GPS (celular do operador) --------------------------
posicao = capturar_posicao(ciclo) if modo_real else None
velocidade_gps, origem_velocidade = (
    obter_velocidade_kmh(posicao) if modo_real else (None, "indisponivel")
)

# ---- Captura de clima (Open-Meteo, a partir da posicao GPS) --------
clima = None
if modo_real and posicao and posicao.get("lat") is not None:
    clima = buscar_clima_atual(posicao["lat"], posicao["lon"])

st.sidebar.markdown("### 📍 Status da telemetria")
if modo_real:
    if posicao is None:
        st.sidebar.warning("Aguardando permissão de localização do navegador…")
    else:
        st.sidebar.caption(
            f"GPS: {posicao['lat']:.5f}, {posicao['lon']:.5f} "
            f"(±{posicao.get('accuracy_m', '?')}m) · leitura #{ciclo}"
        )
    if clima and clima["ok"]:
        st.sidebar.caption("Clima: Open-Meteo · atualizado agora")
    elif clima and not clima["ok"]:
        st.sidebar.error(f"Falha na API de clima: {clima.get('erro', '')}")
else:
    st.sidebar.caption("Modo manual — sem captura em tempo real.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Variáveis ambientais")

umidade_auto = clima["umidade_solo"] if clima and clima.get("umidade_solo") is not None else None
if manual_umidade or umidade_auto is None:
    umidade_solo = st.sidebar.slider("Umidade do solo (%)", 0, 100, int(umidade_auto or 20))
else:
    umidade_solo = umidade_auto
    st.sidebar.metric("Umidade do solo (auto)", f"{umidade_solo:.0f}%")

declividade = st.sidebar.slider(
    "Declividade (%)", 0, 25, 8,
    help="Ainda depende de mapa GIS/satélite por talhão — não coberto pelo GPS do celular."
)

if manual_clima or not (clima and clima["ok"]):
    chuva_24h   = st.sidebar.slider("Chuva 24h (mm)",    0, 40,  4)
    temperatura = st.sidebar.slider("Temperatura (°C)", 15, 45, 26)
else:
    chuva_24h = clima["chuva_24h"]
    temperatura = clima["temperatura"]
    st.sidebar.metric("Chuva 24h (auto)", f"{chuva_24h:.1f} mm")
    st.sidebar.metric("Temperatura (auto)", f"{temperatura:.1f}°C")

st.sidebar.markdown("### Variáveis operacionais")

if manual_velocidade or velocidade_gps is None:
    velocidade = st.sidebar.slider("Velocidade (km/h)", 0, 20, 7)
    if modo_real and not manual_velocidade:
        st.sidebar.caption("⏳ Aguardando 2ª leitura de GPS para calcular velocidade…")
else:
    velocidade = velocidade_gps
    st.sidebar.metric("Velocidade (auto)", f"{velocidade:.1f} km/h", origem_velocidade)

idade_maquina = st.sidebar.slider("Idade da máquina (anos)", 0, 20, 5)

# ---- Horas de operação: contador automático enquanto conectado -----
# Enquanto o GPS está conectado (posicao != None) e o campo não está em
# modo manual, o relógio soma sozinho o tempo decorrido a cada rerun.
# Se o GPS cai ou o modo manual é ligado, o contador PAUSA (não some nem
# volta a correr sozinho a partir do zero) até reconectar.
if "_horas_acumuladas" not in st.session_state:
    st.session_state["_horas_acumuladas"] = 0.0

contando_automatico = modo_real and not manual_horas and posicao is not None
if contando_automatico:
    agora = time.time()
    ultimo_tick = st.session_state.get("_ultimo_tick_horas")
    if ultimo_tick is not None:
        st.session_state["_horas_acumuladas"] += (agora - ultimo_tick) / 3600
    st.session_state["_ultimo_tick_horas"] = agora
else:
    st.session_state.pop("_ultimo_tick_horas", None)  # evita "pulo" ao reconectar

if manual_horas:
    horas_operacao = st.sidebar.slider("Horas de operação (acumuladas)", 0, 14, 5)
    st.sidebar.caption(f"≈ {formatar_hhmm(horas_operacao)}")
    origem_horas = "manual"
else:
    horas_operacao = round(min(st.session_state["_horas_acumuladas"], 14.0), 2)
    origem_horas = "gps_auto" if contando_automatico else "pausado"
    st.sidebar.metric("Horas de operação (auto)", formatar_hhmm(horas_operacao), origem_horas)
    if st.sidebar.button("🔄 Reiniciar contador de horas"):
        st.session_state["_horas_acumuladas"] = 0.0
        st.session_state.pop("_ultimo_tick_horas", None)
        st.rerun()

status_operacao = st.sidebar.selectbox(
    "Tipo de operação", ["Colheita", "Deslocamento", "Manobra"]
)

# =====================================================================
# Carregamento do modelo
# =====================================================================
modelo = carregar_modelo()
if modelo is None:
    st.error(
        f"❌ Modelo não encontrado em `{MODELO_PATH}`. "
        "Execute o notebook `modelo/random_forest.ipynb` primeiro."
    )
    st.stop()

# =====================================================================
# Predicao em tempo real + override de MAX
# =====================================================================
entrada_dict = {
    "umidade_solo":    umidade_solo,
    "declividade":     declividade,
    "chuva_24h":       chuva_24h,
    "temperatura":     temperatura,
    "velocidade":      velocidade,
    "idade_maquina":   idade_maquina,
    "horas_operacao":  horas_operacao,
    "status_operacao": status_operacao,
}
entrada = pd.DataFrame([entrada_dict])

# Predicao do modelo
classe_modelo = modelo.predict(entrada)[0]
probas        = modelo.predict_proba(entrada)[0]
classes_modelo = modelo.classes_

# Override: qualquer variavel no MAX -> ALTO imediato
em_max, vars_max = verificar_max(entrada_dict)
n_alertas, vars_alerta = contar_alertas(entrada_dict)

if em_max:
    classe_pred = "ALTO"
    motivo_override = f"Variável(is) no limite máximo: {', '.join(vars_max)}"
else:
    classe_pred = classe_modelo
    motivo_override = None

score        = float(probas.max() * 100)
recomendacao = gerar_recomendacao(entrada_dict, classe_pred, vars_max)

# =====================================================================
# Persistência automática — tabela local (SQLite) + Oracle (best effort)
# =====================================================================
# Throttle: evita gravar a cada rerun causado por arraste de slider; só
# grava de fato a cada ~5s (mesmo ritmo do ciclo de telemetria).
INTERVALO_MIN_GRAVACAO = 10  # segundos — mesmo ritmo do ciclo de telemetria
agora_ts = time.time()
pode_gravar = (
    modo_real
    and (agora_ts - st.session_state.get("_ultimo_save", 0) >= INTERVALO_MIN_GRAVACAO)
)

if pode_gravar:
    idx_probs = {c: float(p) for c, p in zip(classes_modelo, probas)}
    registro = {
        **entrada_dict,
        "classe_risco": classe_pred,
        "score_risco": score,
        "recomendacao": recomendacao,
        "prob_baixo": idx_probs.get("BAIXO"),
        "prob_medio": idx_probs.get("MEDIO"),
        "prob_alto": idx_probs.get("ALTO"),
        "fonte_velocidade": "manual" if manual_velocidade else origem_velocidade,
        "fonte_clima": "manual" if manual_clima else "open-meteo",
        "fonte_horas": origem_horas,
    }
    gravou_oracle = registrar_oracle(CONFIG_ORACLE, registro)
    registrar_local(registro, gravado_oracle=gravou_oracle)
    st.session_state["_ultimo_save"] = agora_ts

# =====================================================================
# Header
# =====================================================================
st.title("🌾 Sompo AgroPredict — Dashboard de Risco")
st.markdown(
    "**Em vez de pagar o sinistro depois, calculamos o risco antes.** "
    "Plataforma de monitoramento preventivo de risco operacional em equipamentos agrícolas."
)
st.markdown("---")

# =====================================================================
# Resultado em destaque
# =====================================================================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(
        f"""
        <div style='background-color:{CORES_RISCO[classe_pred]};padding:30px;
                    border-radius:12px;color:white;text-align:center;'>
            <h1 style='margin:0;font-size:60px;'>{EMOJI_RISCO[classe_pred]} {classe_pred}</h1>
            <h2 style='margin:5px 0;'>Score: {score:.1f} / 100</h2>
            <p style='margin:10px 0 0 0;font-size:18px;'><strong>{recomendacao}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if motivo_override:
        st.warning(f"⚡ **Override ativado:** {motivo_override}")
    if n_alertas > 0 and not em_max:
        st.info(f"⚠️ {n_alertas} variável(is) em alerta: {', '.join(vars_alerta)}")

with col2:
    st.markdown("### Probabilidades")
    for c, p in zip(classes_modelo, probas):
        st.markdown(f"**{EMOJI_RISCO[c]} {c}**")
        st.progress(float(p))
        st.caption(f"{p*100:.1f}%")

with col3:
    st.markdown("### Contexto")
    st.metric("Operação",      status_operacao)
    st.metric("Velocidade",    f"{velocidade} km/h")
    st.metric("Idade máquina", f"{idade_maquina} anos")

# =====================================================================
# Detalhes da entrada
# =====================================================================
st.markdown("---")
st.subheader("📊 Detalhes da operação avaliada")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Umidade do solo", f"{umidade_solo}%",  label_alerta(umidade_solo, "umidade_solo"))
col_b.metric("Declividade",     f"{declividade}%",   label_alerta(declividade,  "declividade"))
col_c.metric("Chuva 24h",       f"{chuva_24h} mm",   label_alerta(chuva_24h,    "chuva_24h"))
col_d.metric("Temperatura",     f"{temperatura}°C",  label_alerta(temperatura,  "temperatura"))

col_e, col_f, col_g, col_h = st.columns(4)
col_e.metric("Velocidade",      f"{velocidade} km/h",    label_alerta(velocidade,     "velocidade"))
col_f.metric("Idade máquina",   f"{idade_maquina} anos", label_alerta(idade_maquina,  "idade_maquina"))
col_g.metric("Horas operação",  formatar_hhmm(horas_operacao),    label_alerta(horas_operacao, "horas_operacao"))
col_h.metric("Tipo",            status_operacao)

# =====================================================================
# Feature Importance
# =====================================================================
st.markdown("---")
st.subheader("🔍 Por que essa decisão? — Feature Importance")
st.markdown(
    "O modelo Random Forest indica qual variável mais pesou na decisão. "
    "Essa transparência é essencial para uma seguradora justificar suas avaliações."
)

clf    = modelo.named_steps["clf"]
preproc = modelo.named_steps["preproc"]
cols_num = ["umidade_solo","declividade","chuva_24h","temperatura",
            "velocidade","idade_maquina","horas_operacao"]
nomes_features = (
    cols_num +
    list(preproc.named_transformers_["cat"].get_feature_names_out(["status_operacao"]))
)
df_imp = pd.DataFrame({
    "feature":     nomes_features,
    "importancia": clf.feature_importances_
}).sort_values("importancia", ascending=True)

st.bar_chart(df_imp.set_index("feature"), horizontal=True)

# =====================================================================
# Historico
# =====================================================================
st.markdown("---")
st.subheader("📅 Histórico de predições")

tab_local, tab_oracle = st.tabs(["💾 Local (SQLite — sempre ativo)", "🗄️ Oracle XE"])

with tab_local:
    total_local = contar_local()
    st.success(f"✅ Tabela local: **{total_local}** predições registradas em `dashboard/data/historico_local.db`.")
    st.caption(
        "Essa tabela cresce sozinha a cada ciclo de telemetria (≈5s) e não depende do Oracle. "
        "É gravada localmente, no disco de quem está rodando o dashboard."
    )
    if total_local > 0:
        df_local = carregar_local(limite=200)
        st.dataframe(df_local, use_container_width=True, height=320)
    if not modo_real:
        st.info("Ative o modo tempo real na barra lateral para a gravação automática funcionar.")

with tab_oracle:
    historico = buscar_historico_oracle()
    if historico is not None and len(historico) > 0:
        st.success(f"✅ Conectado ao Oracle XE. {len(historico)} predições no histórico.")
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            dist = historico["CLASSE_RISCO"].value_counts()
            st.markdown("**Distribuição histórica:**")
            for c in ["BAIXO", "MEDIO", "ALTO"]:
                qtd = int(dist.get(c, 0))
                st.markdown(f"{EMOJI_RISCO[c]} **{c}**: {qtd}")
        with col_h2:
            st.dataframe(historico.head(20), use_container_width=True)
    else:
        st.info(
            "ℹ️ Sem conexão com Oracle XE agora. As predições continuam sendo gravadas "
            "normalmente na tabela local (aba ao lado) — nada é perdido."
        )
        if "historico_sessao" not in st.session_state:
            st.session_state.historico_sessao = []

        if st.button("💾 Adicionar predição atual ao histórico da sessão"):
            st.session_state.historico_sessao.append({
                "Data/Hora":    datetime.now().strftime("%H:%M:%S"),
                "Classe":       classe_pred,
                "Score":        round(score, 1),
                "Alertas":      n_alertas,
                "MAX ativado":  "Sim" if em_max else "Não",
                "Recomendação": recomendacao,
                "Operação":     status_operacao,
            })

        if st.session_state.historico_sessao:
            st.dataframe(
                pd.DataFrame(st.session_state.historico_sessao),
                use_container_width=True
            )

# =====================================================================
# Footer
# =====================================================================
st.markdown("---")
st.caption(
    f"Sompo AgroPredict · Sprint 2 · Challenge Sompo × FIAP · "
    f"Modelo Random Forest · {datetime.now().year}"
)