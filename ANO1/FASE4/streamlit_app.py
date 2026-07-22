"""
FarmTech Solutions — Dashboard Agrícola Inteligente
Streamlit • Scikit-Learn • Regressão Supervisionada
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from farmtech_ml import (
    load_data,
    train_and_evaluate,
    treat_outliers_iqr,
    suggest_actions,
    get_feature_importance,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    IRRIG_FEATURES_NUM, IRRIG_FEATURES_CAT,
    FERT_FEATURES_NUM,  FERT_FEATURES_CAT,
    TARGET,
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FarmTech Solutions — IA Agrícola",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0f1a0f; }
[data-testid="stSidebar"] { background-color: #1a2e1a; }
h1, h2, h3 { color: #6fcf6f !important; }
[data-testid="metric-container"] {
    background-color: #1e3120; border: 1px solid #3a5c3a;
    border-radius: 10px; padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #a8d5a2 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #6fcf6f !important; font-size: 1.5rem !important;
}
.stTabs [data-baseweb="tab-list"] { background-color: #1a2e1a; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #a8d5a2; }
.stTabs [aria-selected="true"] { background-color: #2d5c2d !important; color: #6fcf6f !important; }
.sugestao-box {
    background-color: #1e3120; border-left: 4px solid #6fcf6f;
    border-radius: 8px; padding: 14px 18px; margin: 6px 0;
    color: #d4edd4; font-size: 0.97rem;
}
.doc-box {
    background-color: #162416; border: 1px solid #3a5c3a;
    border-radius: 8px; padding: 16px 20px; margin: 8px 0; color: #c8e6c8;
}
hr { border-color: #3a5c3a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DADOS & MODELOS (cache)
# ─────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_data("ANO1/FASE4/dataset_agricola.csv")
    for col in ["temperatura", "umidade", "ph", "chuva", "rendimento"]:
        df = treat_outliers_iqr(df, col)
    df["indice_fertilizacao"] = df["N"] + df["P"] + df["K"]
    return df

@st.cache_resource
def get_all_models():
    df = load_data("dataset_agricola.csv")
    return train_and_evaluate(df)

df = get_data()
res_rend, res_irrig, res_fert = get_all_models()

best_rend_name  = max(res_rend,  key=lambda k: res_rend[k]["R²"])
best_irrig_name = max(res_irrig, key=lambda k: res_irrig[k]["R²"])
best_fert_name  = max(res_fert,  key=lambda k: res_fert[k]["R²"])

pipe_rend  = res_rend[best_rend_name]["pipeline"]
pipe_irrig = res_irrig[best_irrig_name]["pipeline"]
pipe_fert  = res_fert[best_fert_name]["pipeline"]

PALETTE = {"cafe": "#6fcf6f", "soja": "#a8d5a2", "milho": "#3a8c3a", "trigo": "#e0b060"}
IRR_PAL = {"LIGADA": "#6fcf6f", "DESLIGADA": "#c0392b"}

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 FarmTech Solutions")
    st.markdown("**Assistente Agrícola Inteligente**")
    st.divider()

    st.markdown("### 📌 Filtros do Dataset")
    culturas_sel  = st.multiselect("Cultura",   df["cultura"].unique().tolist(),   default=df["cultura"].unique().tolist())
    irrigacao_sel = st.multiselect("Irrigação", df["irrigacao"].unique().tolist(), default=df["irrigacao"].unique().tolist())
    df_f = df[df["cultura"].isin(culturas_sel) & df["irrigacao"].isin(irrigacao_sel)]

    st.divider()
    st.markdown(f"**Dataset filtrado:** {len(df_f)} registros")
    st.markdown(f"**Melhor modelo (rendimento):** {best_rend_name} · R²={res_rend[best_rend_name]['R²']:.4f}")
    st.markdown(f"**Melhor modelo (irrigação):**  {best_irrig_name} · R²={res_irrig[best_irrig_name]['R²']:.4f}")
    st.markdown(f"**Melhor modelo (fertilização):** {best_fert_name} · R²={res_fert[best_fert_name]['R²']:.4f}")

    st.divider()
    st.markdown("### 🔮 Simular Previsão")
    sim_cultura   = st.selectbox("Cultura", df["cultura"].unique())
    sim_irrigacao = st.selectbox("Irrigação", df["irrigacao"].unique())
    sim_N    = st.slider("Nitrogênio (N)",   0, 1, 1)
    sim_P    = st.slider("Fósforo (P)",      0, 1, 1)
    sim_K    = st.slider("Potássio (K)",     0, 1, 1)
    sim_temp = st.slider("Temperatura (°C)", 15.0, 45.0, 25.0, 0.5)
    sim_umid = st.slider("Umidade (%)",      10.0, 100.0, 60.0, 1.0)
    sim_ph   = st.slider("pH",               4.5,  7.5,   6.0,  0.05)
    sim_chuva= st.slider("Chuva (mm)",       0.0,  250.0, 130.0, 5.0)

    # Previsão de rendimento
    inp_rend = pd.DataFrame(
        [[sim_N, sim_P, sim_K, sim_temp, sim_umid, sim_ph, sim_chuva, sim_cultura, sim_irrigacao]],
        columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    )
    rend_prev = pipe_rend.predict(inp_rend)[0]

    # Previsão de volume de irrigação
    inp_irrig = pd.DataFrame(
        [[sim_N, sim_P, sim_K, sim_temp, sim_umid, sim_ph, rend_prev, sim_cultura, sim_irrigacao]],
        columns=IRRIG_FEATURES_NUM + IRRIG_FEATURES_CAT,
    )
    irrig_prev = max(0, pipe_irrig.predict(inp_irrig)[0])

    # Previsão de índice de fertilização
    inp_fert = pd.DataFrame(
        [[sim_temp, sim_umid, sim_ph, sim_chuva, rend_prev, sim_cultura, sim_irrigacao]],
        columns=FERT_FEATURES_NUM + FERT_FEATURES_CAT,
    )
    fert_prev = np.clip(pipe_fert.predict(inp_fert)[0], 0, 3)

    acoes = suggest_actions(sim_umid, sim_ph, rend_prev, irrig_prev, fert_prev, sim_cultura)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rendimento (kg)", f"{rend_prev:.1f}")
    c2.metric("Irrigação (mm)",  f"{irrig_prev:.0f}")
    c3.metric("Fert. (0–3)",     f"{fert_prev:.2f}")

# ─────────────────────────────────────────────
#  CABEÇALHO
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🌾 FarmTech Solutions</h1>"
    "<p style='text-align:center; color:#a8d5a2; font-size:1.1rem;'>"
    "Assistente Agrícola Inteligente · Regressão Supervisionada · Scikit-Learn</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral",
    "🔬 Análise Exploratória",
    "🤖 Modelos de ML",
    "🔮 Previsão & Ações",
    "📋 Dados Brutos",
    "📖 Documentação",
])

# ════════════════════════════════════════════
#  TAB 1 — VISÃO GERAL
# ════════════════════════════════════════════
with tab1:
    st.subheader("Indicadores Gerais do Campo")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total de Registros",   len(df_f))
    c2.metric("Rendimento Médio (kg)", f"{df_f['rendimento'].mean():.1f}")
    c3.metric("Umidade Média (%)",     f"{df_f['umidade'].mean():.1f}")
    c4.metric("pH Médio",              f"{df_f['ph'].mean():.2f}")
    c5.metric("Chuva Média (mm)",      f"{df_f['chuva'].mean():.1f}")
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            df_f.groupby("cultura")["rendimento"].mean().reset_index(),
            x="cultura", y="rendimento",
            color="cultura",
            color_discrete_map=PALETTE,
            title="Rendimento Médio por Cultura (kg)",
            template="plotly_dark",
        )
        fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.box(
            df_f, x="irrigacao", y="rendimento", color="irrigacao",
            color_discrete_map=IRR_PAL,
            title="Rendimento por Status de Irrigação",
            template="plotly_dark",
        )
        fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.pie(
            df_f["irrigacao"].value_counts().reset_index(),
            names="irrigacao", values="count",
            color_discrete_sequence=["#6fcf6f","#c0392b"],
            title="Irrigação Ativa vs Desligada",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        fig = px.pie(
            df_f["cultura"].value_counts().reset_index(),
            names="cultura", values="count",
            color_discrete_sequence=list(PALETTE.values()),
            title="Distribuição das Culturas",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
#  TAB 2 — EDA
# ════════════════════════════════════════════
with tab2:
    st.subheader("Análise Exploratória dos Dados")

    col1, col2 = st.columns(2)
    with col1:
        corr = df_f[NUMERIC_FEATURES + [TARGET]].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=np.round(corr.values,2), texttemplate="%{text}",
        ))
        fig.update_layout(title="Matriz de Correlação", template="plotly_dark",
                          plot_bgcolor="#1e3120", paper_bgcolor="#1e3120", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for cult, grp in df_f.groupby("cultura"):
            fig.add_trace(go.Scatter(
                x=grp["umidade"], y=grp["rendimento"], mode="markers",
                name=cult, marker=dict(color=PALETTE.get(cult,"#6fcf6f"), opacity=0.6, size=6),
            ))
            z = np.polyfit(grp["umidade"], grp["rendimento"], 1)
            x_l = np.linspace(grp["umidade"].min(), grp["umidade"].max(), 50)
            fig.add_trace(go.Scatter(
                x=x_l, y=np.poly1d(z)(x_l), mode="lines",
                line=dict(color=PALETTE.get(cult,"#6fcf6f"), dash="dash", width=1.5),
                showlegend=False,
            ))
        fig.update_layout(title="Umidade × Rendimento por Cultura",
                          xaxis_title="Umidade (%)", yaxis_title="Rendimento (kg)",
                          template="plotly_dark", plot_bgcolor="#1e3120",
                          paper_bgcolor="#1e3120", height=420)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure()
        for irr, grp in df_f.groupby("irrigacao"):
            fig.add_trace(go.Scatter(
                x=grp["ph"], y=grp["rendimento"], mode="markers",
                name=irr, marker=dict(color=IRR_PAL.get(irr,"#6fcf6f"), opacity=0.6, size=6),
            ))
            z = np.polyfit(grp["ph"], grp["rendimento"], 1)
            x_l = np.linspace(grp["ph"].min(), grp["ph"].max(), 50)
            fig.add_trace(go.Scatter(
                x=x_l, y=np.poly1d(z)(x_l), mode="lines",
                line=dict(color=IRR_PAL.get(irr,"#6fcf6f"), dash="dash", width=1.5),
                showlegend=False,
            ))
        fig.update_layout(title="pH × Rendimento",
                          xaxis_title="pH do Solo", yaxis_title="Rendimento (kg)",
                          template="plotly_dark", plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df_f, x="rendimento", nbins=30, color="cultura", barmode="overlay",
            color_discrete_map=PALETTE,
            title="Distribuição do Rendimento por Cultura",
            template="plotly_dark", labels={"rendimento": "Rendimento (kg)"},
        )
        fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detecção de Outliers (Boxplot IQR)")
    col_sel = st.selectbox("Variável:", NUMERIC_FEATURES + [TARGET])
    fig = px.box(df_f, y=col_sel, color="cultura", color_discrete_map=PALETTE,
                 template="plotly_dark", title=f"Boxplot — {col_sel}")
    fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
#  TAB 3 — MODELOS DE ML
# ════════════════════════════════════════════
with tab3:
    st.subheader("Comparação dos Modelos de Regressão Supervisionada")

    target_choice = st.radio(
        "Selecione o alvo:",
        ["Rendimento (kg)", "Volume de Irrigação (mm)", "Índice de Fertilização (0–3)"],
        horizontal=True,
    )
    res_map = {
        "Rendimento (kg)":               (res_rend,  best_rend_name),
        "Volume de Irrigação (mm)":      (res_irrig, best_irrig_name),
        "Índice de Fertilização (0–3)":  (res_fert,  best_fert_name),
    }
    res_cur, best_cur = res_map[target_choice]

    # ── Tabela com MAE, MSE, RMSE, R² ──
    rows = []
    for name, m in res_cur.items():
        rows.append({
            "Modelo": name,
            "MAE":  m["MAE"],
            "MSE":  m["MSE"],
            "RMSE": m["RMSE"],
            "R²":   m["R²"],
            "":     "⭐ Melhor" if name == best_cur else "",
        })
    df_met = pd.DataFrame(rows).sort_values("R²", ascending=False)
    st.dataframe(
        df_met.style
            .background_gradient(subset=["R²"], cmap="Greens")
            .background_gradient(subset=["MAE","MSE","RMSE"], cmap="Reds_r")
            .format({"MAE":"{:.3f}","MSE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.caption(
        "**MAE** = Erro Médio Absoluto · **MSE** = Erro Quadrático Médio · "
        "**RMSE** = Raiz do MSE · **R²** = Coeficiente de Determinação (1.0 = perfeito)"
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig = px.bar(df_met, x="Modelo", y="R²", color="R²",
                     color_continuous_scale="Greens",
                     title="R² por Modelo (↑ melhor)", template="plotly_dark")
        fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

    with col_m2:
        fig = px.bar(df_met, x="Modelo", y="RMSE", color="RMSE",
                     color_continuous_scale="Reds_r",
                     title="RMSE por Modelo (↓ melhor)", template="plotly_dark")
        fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

    # ── MSE lado a lado ──
    fig_mse = px.bar(df_met, x="Modelo", y="MSE", color="MSE",
                     color_continuous_scale="Oranges_r",
                     title="MSE por Modelo (↓ melhor)", template="plotly_dark")
    fig_mse.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
    st.plotly_chart(fig_mse, use_container_width=True)

    st.divider()
    # ── Previsto × Real ──
    sel_model = st.selectbox("Inspecionar modelo:", list(res_cur.keys()))
    r = res_cur[sel_model]

    # ── Melhores hiperparâmetros (DT e RF) ──────────────────────────────
    if sel_model in ("Árvore de Decisão", "Random Forest") and r.get("best_params"):
        st.markdown("**🎯 Melhores hiperparâmetros encontrados pelo RandomizedSearchCV:**")
        # Limpa o prefixo "model__" para exibição
        params_clean = {k.replace("model__", ""): v for k, v in r["best_params"].items()}
        cols_p = st.columns(len(params_clean))
        for col_p, (k, v) in zip(cols_p, params_clean.items()):
            col_p.metric(k, v)
        st.caption(
            "RandomizedSearchCV: cv=3 folds · n_jobs=-1 (todos os núcleos) · "
            "n_iter=50 combinações aleatórias · random_state=42"
        )

    col_pv, col_res = st.columns(2)

    with col_pv:
        lim = [min(r["y_test"].min(), r["y_pred"].min()),
               max(r["y_test"].max(), r["y_pred"].max())]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r["y_test"], y=r["y_pred"], mode="markers",
            marker=dict(color="#6fcf6f", opacity=0.7, size=7), name="Previsto vs Real",
        ))
        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="#ff6b6b", dash="dash"), name="Perfeito",
        ))
        fig.update_layout(
            title=f"Previsto × Real — {sel_model}",
            xaxis_title="Valor Real", yaxis_title="Valor Previsto",
            template="plotly_dark", plot_bgcolor="#1e3120", paper_bgcolor="#1e3120",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_res:
        residuos = np.array(r["y_pred"]) - np.array(r["y_test"])
        fig = px.histogram(
            x=residuos, nbins=30, title=f"Resíduos — {sel_model}",
            labels={"x": "Resíduo (Previsto − Real)"},
            color_discrete_sequence=["#6fcf6f"], template="plotly_dark",
        )
        fig.update_layout(plot_bgcolor="#1e3120", paper_bgcolor="#1e3120")
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance (Random Forest) ──
    st.divider()
    st.subheader("📌 Importância de Features — Random Forest")

    fi_map = {
        "Rendimento (kg)":              (res_rend["Random Forest"]["pipeline"],  NUMERIC_FEATURES,  CATEGORICAL_FEATURES),
        "Volume de Irrigação (mm)":     (res_irrig["Random Forest"]["pipeline"], IRRIG_FEATURES_NUM, IRRIG_FEATURES_CAT),
        "Índice de Fertilização (0–3)": (res_fert["Random Forest"]["pipeline"],  FERT_FEATURES_NUM,  FERT_FEATURES_CAT),
    }
    pipe_fi, num_fi, cat_fi = fi_map[target_choice]
    df_imp = get_feature_importance(pipe_fi, num_fi, cat_fi)

    if not df_imp.empty:
        fig = px.bar(
            df_imp.head(12), x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Greens",
            title=f"Top Features — {target_choice}",
            template="plotly_dark",
        )
        fig.update_layout(
            plot_bgcolor="#1e3120", paper_bgcolor="#1e3120",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Feature importance do Random Forest: quanto cada variável contribuiu para as previsões.")

# ════════════════════════════════════════════
#  TAB 4 — PREVISÃO & AÇÕES
# ════════════════════════════════════════════
with tab4:
    st.subheader("🔮 Previsão Completa & Sugestões de Manejo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendimento Previsto (kg)", f"{rend_prev:.2f}",
              delta=f"{rend_prev - df['rendimento'].mean():.1f} vs média")
    c2.metric("Volume Irrigação Previsto (mm)", f"{irrig_prev:.0f}",
              delta=f"{irrig_prev - df['chuva'].mean():.0f} vs média")
    c3.metric("Índice Fertilização (0–3)", f"{fert_prev:.2f}",
              delta=f"{fert_prev - df['indice_fertilizacao'].mean():.2f} vs média")

    st.caption(
        f"Rendimento: **{best_rend_name}** (R²={res_rend[best_rend_name]['R²']:.4f}) · "
        f"Irrigação: **{best_irrig_name}** (R²={res_irrig[best_irrig_name]['R²']:.4f}) · "
        f"Fertilização: **{best_fert_name}** (R²={res_fert[best_fert_name]['R²']:.4f})"
    )

    st.divider()
    # ── Três gauges ──
    g1, g2, g3 = st.columns(3)

    def gauge(value, title, max_val, ref, suffix=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"color": "#6fcf6f", "size": 13}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#6fcf6f"},
                "bar": {"color": "#6fcf6f"},
                "steps": [
                    {"range": [0, max_val*0.33], "color": "#4a1010"},
                    {"range": [max_val*0.33, max_val*0.66], "color": "#3a4a10"},
                    {"range": [max_val*0.66, max_val], "color": "#1a3a1a"},
                ],
                "threshold": {"line": {"color": "#ff6b6b","width":3},
                              "thickness": 0.8, "value": ref},
            },
            number={"font": {"color":"#6fcf6f"}, "suffix": suffix},
        ))
        fig.update_layout(paper_bgcolor="#1e3120", height=240, margin=dict(t=50,b=10,l=20,r=20))
        return fig

    with g1:
        st.plotly_chart(gauge(rend_prev, "Rendimento (kg)", 250, df["rendimento"].mean(), " kg"),
                        use_container_width=True)
    with g2:
        st.plotly_chart(gauge(irrig_prev, "Irrigação (mm)", 250, df["chuva"].mean(), " mm"),
                        use_container_width=True)
    with g3:
        st.plotly_chart(gauge(fert_prev, "Fertilização (0–3)", 3, df["indice_fertilizacao"].mean()),
                        use_container_width=True)

    st.divider()
    st.subheader("📋 Recomendações de Manejo Geradas pela IA")
    st.caption("Ajuste os parâmetros na sidebar para simular diferentes cenários.")
    for acao in acoes:
        st.markdown(f"<div class='sugestao-box'>{acao}</div>", unsafe_allow_html=True)

    st.divider()
    # ── Comparativo ──
    media_cult = df[df["cultura"]==sim_cultura]["rendimento"].mean()
    fig = go.Figure(go.Bar(
        x=["Média Global", f"Média {sim_cultura.capitalize()}", "Previsão Atual"],
        y=[df["rendimento"].mean(), media_cult, rend_prev],
        marker_color=["#3a8c3a","#6fcf6f","#a8d5a2"],
        text=[f"{df['rendimento'].mean():.1f}", f"{media_cult:.1f}", f"{rend_prev:.1f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Comparativo de Rendimento: Previsão vs Médias Históricas",
        yaxis_title="Rendimento (kg)", template="plotly_dark",
        plot_bgcolor="#1e3120", paper_bgcolor="#1e3120", yaxis_range=[0, 280],
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
#  TAB 5 — DADOS BRUTOS
# ════════════════════════════════════════════
with tab5:
    st.subheader("📋 Dataset Agrícola")
    st.caption(f"{len(df_f)} registros com os filtros atuais.")

    search = st.text_input("🔍 Buscar:", "")
    df_show = df_f[df_f.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)] \
              if search else df_f

    st.dataframe(
        df_show.style.background_gradient(subset=["rendimento"], cmap="Greens"),
        use_container_width=True, height=420,
    )

    csv_bytes = df_f.to_csv(index=False).encode()
    st.download_button("⬇️ Exportar CSV Filtrado", data=csv_bytes,
                       file_name="farmtech_filtrado.csv", mime="text/csv")
    st.markdown(
        f"mín={df_f['rendimento'].min():.1f} kg · "
        f"máx={df_f['rendimento'].max():.1f} kg · "
        f"σ={df_f['rendimento'].std():.1f} kg"
    )

# ════════════════════════════════════════════
#  TAB 6 — DOCUMENTAÇÃO DO PROCESSO
# ════════════════════════════════════════════
with tab6:
    st.subheader("📖 Documentação do Pipeline de Machine Learning")

    st.markdown("""
<div class='doc-box'>
<h3 style='color:#6fcf6f'>1. Objetivo</h3>
Construir três modelos de regressão supervisionada capazes de prever:<br>
• <b>Rendimento da cultura</b> (kg) — variável-alvo principal<br>
• <b>Volume de irrigação necessário</b> (mm) — suporte à tomada de decisão hídrica<br>
• <b>Índice de fertilização</b> (0–3) — demanda combinada de N, P e K
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='doc-box'>
<h3 style='color:#6fcf6f'>2. Dataset</h3>
<b>500 registros</b> de sensores IoT simulados com as variáveis:<br>
<code>N, P, K</code> (nutrientes binários) · <code>temperatura</code> · <code>umidade</code> · 
<code>ph</code> · <code>chuva</code> · <code>cultura</code> · <code>irrigacao</code> · <code>rendimento</code>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='doc-box'>
<h3 style='color:#6fcf6f'>3. Limpeza e Pré-processamento</h3>
• <b>Remoção de duplicatas</b><br>
• <b>Tratamento de outliers via IQR</b>: valores fora do intervalo [Q1 − 1,5·IQR, Q3 + 1,5·IQR] 
  são substituídos pela mediana<br>
• <b>StandardScaler</b> para variáveis numéricas (média=0, σ=1)<br>
• <b>OneHotEncoder</b> para variáveis categóricas (<code>cultura</code>, <code>irrigacao</code>) 
  — evita ordenação implícita e multicolinearidade
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='doc-box'>
<h3 style='color:#6fcf6f'>4. Modelos Avaliados</h3>

| Modelo | Tipo | Característica |
|---|---|---|
| Regressão Linear | Linear | Simples, interpretável, baseline |
| Ridge Regression | Linear regularizado | Evita overfitting via penalidade L2 |
| Árvore de Decisão | Não linear | Alta interpretabilidade, tendência a overfitting |
| Random Forest | Não linear (ensemble) | Robusto, fornece importância de features |
| Gradient Boosting | Não linear (boosting) | Alta performance, aprende erros sequencialmente |
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='doc-box'>
<h3 style='color:#6fcf6f'>5. Métricas de Avaliação</h3>

| Métrica | Fórmula | Interpretação |
|---|---|---|
| <b>MAE</b> | mean(|y − ŷ|) | Erro médio em unidade original; robusto a outliers |
| <b>MSE</b> | mean((y − ŷ)²) | Penaliza erros grandes; sensível a outliers |
| <b>RMSE</b> | √MSE | Mesma unidade do target; mais intuitivo que MSE |
| <b>R²</b> | 1 − SS_res/SS_tot | Proporção da variância explicada (1,0 = perfeito) |

<br><b>Resultado destaque</b>: Regressão Linear alcançou R²=0.976 no rendimento — 
indica que as variáveis do sensor explicam ~97,6% da variação do rendimento.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='doc-box'>
<h3 style='color:#6fcf6f'>6. Decisões Técnicas Justificadas</h3>
• <b>Por que StandardScaler?</b> Modelos como Regressão Linear e Ridge são sensíveis à escala. 
  Normalizar garante que temperatura (15–45) não domine umidade (0–100).<br>
• <b>Por que OneHotEncoder em vez de LabelEncoder?</b> LabelEncoder criaria ordem implícita 
  (café=0 < soja=1 < trigo=2), o que é incorreto para variáveis nominais.<br>
• <b>Por que 80/20 train-test split?</b> Padrão consolidado; com 500 registros, 100 amostras 
  de teste são suficientes para estimar a generalização.<br>
• <b>Por que Random Forest com 150 árvores?</b> Equilibrio entre performance e tempo de treino; 
  valores abaixo de 100 aumentam a variância das previsões.<br>
• <b>Volume de irrigação</b>: usamos <code>chuva</code> como proxy do volume hídrico necessário, 
  preditores incluem umidade, temperatura, pH e rendimento esperado.
</div>
""", unsafe_allow_html=True)

    # Tabela resumo de todos os modelos e targets
    st.divider()
    st.subheader("Resumo Completo de Métricas — Todos os Modelos e Alvos")
    all_rows = []
    for alvo, res in [("Rendimento", res_rend), ("Irrigação", res_irrig), ("Fertilização", res_fert)]:
        for name, m in res.items():
            all_rows.append({"Alvo": alvo, "Modelo": name,
                             "MAE": m["MAE"], "MSE": m["MSE"],
                             "RMSE": m["RMSE"], "R²": m["R²"]})
    df_all = pd.DataFrame(all_rows)
    st.dataframe(
        df_all.style
            .background_gradient(subset=["R²"], cmap="Greens")
            .background_gradient(subset=["RMSE"], cmap="Reds_r")
            .format({"MAE":"{:.3f}","MSE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

# ─────────────────────────────────────────────
#  RODAPÉ
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#3a5c3a; font-size:0.85rem;'>"
    "FarmTech Solutions · Agricultura Cognitiva · "
    "Powered by Scikit-Learn & Streamlit</p>",
    unsafe_allow_html=True,
)
