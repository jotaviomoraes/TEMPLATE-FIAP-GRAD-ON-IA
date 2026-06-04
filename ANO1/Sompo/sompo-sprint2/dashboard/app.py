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
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

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
st.sidebar.markdown("### Variáveis ambientais")

umidade_solo = st.sidebar.slider("Umidade do solo (%)",  0, 100, 20)
declividade  = st.sidebar.slider("Declividade (%)",       0,  25,  8)
chuva_24h    = st.sidebar.slider("Chuva 24h (mm)",        0,  40,  4)
temperatura  = st.sidebar.slider("Temperatura (°C)",     15,  45, 26)

st.sidebar.markdown("### Variáveis operacionais")
velocidade      = st.sidebar.slider("Velocidade (km/h)",              0, 20,  7)
idade_maquina   = st.sidebar.slider("Idade da máquina (anos)",        0, 20,  5)
horas_operacao  = st.sidebar.slider("Horas de operação (acumuladas)", 0, 14,  5)
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
col_g.metric("Horas operação",  f"{horas_operacao}h",    label_alerta(horas_operacao, "horas_operacao"))
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
        "ℹ️ Sem conexão com Oracle XE. Configure o banco (ver `sql/schema.sql`) "
        "e rode `sql/inserir_predicoes.py` para ativar histórico persistente."
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
