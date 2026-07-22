import streamlit as st
import pandas as pd

# Importações do projeto
from carregar_dados import carregar_dados
from ingestao import limpar_dados
from modelo import treinar_modelo, salvar_modelo
from evolucao import avaliar_modelo
from visualizacao import gerar_grafico


# ============================
# CONFIGURAÇÃO DA PÁGINA
# ============================
st.set_page_config(
    page_title="Sistema Preditivo de Irrigação",
    layout="wide"
)


# ============================
# FUNÇÃO PRINCIPAL
# ============================
def main():

    st.title("🌱 Sistema Preditivo de Irrigação")
    st.markdown("Dashboard com Machine Learning para apoio à decisão agrícola")

    st.markdown("---")

    # ============================
    # CARREGAR DADOS
    # ============================
    st.header("📂 1. Carregamento de Dados")

    dados = carregar_dados()

    st.subheader("Visualização da base de dados")
    st.dataframe(dados)

    # ============================
    # ANÁLISE E LIMPEZA
    # ============================
    st.header("🧹 2. Tratamento de Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Estatísticas")
        st.write(dados.describe())

    with col2:
        st.subheader("⚠️ Valores Ausentes")
        st.write(dados.isnull().sum())

    dados_limpos = limpar_dados(dados)

    st.subheader("Dados após limpeza")
    st.dataframe(dados_limpos)

    # ============================
    # TREINAMENTO
    # ============================
    st.header("🤖 3. Treinamento do Modelo")

    modelo, x_teste, y_teste = treinar_modelo(dados_limpos)
    salvar_modelo(modelo)

    st.success("✅ Modelo treinado e salvo com sucesso!")

    # ============================
    # AVALIAÇÃO
    # ============================
    st.header("📊 4. Avaliação do Modelo")

    resultados, previsoes = avaliar_modelo(modelo, x_teste, y_teste)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{resultados['MAE']:.2f}")
    col2.metric("MSE", f"{resultados['MSE']:.2f}")
    col3.metric("RMSE", f"{resultados['RMSE']:.2f}")
    col4.metric("R²", f"{resultados['R2']:.4f}")

    # ============================
    # GRÁFICO
    # ============================
    st.header("📈 5. Visualização")

    fig = gerar_grafico(y_teste, previsoes)
    st.pyplot(fig)

    # ============================
    # PREVISÃO INTERATIVA
    # ============================
    st.header("🔮 6. Previsão em Tempo Real")

    st.sidebar.header("⚙️ Insira os dados")

    umidade = st.sidebar.slider("Umidade (%)", 0, 100, 60)
    temperatura = st.sidebar.slider("Temperatura (°C)", 0, 40, 25)
    chuva = st.sidebar.slider("Chuva (mm)", 0, 50, 10)
    fertilizante = st.sidebar.slider("Fertilizante", 50, 300, 150)

    if st.sidebar.button("Fazer Previsão"):

        entrada = pd.DataFrame({
            "umidade": [umidade],
            "temperatura": [temperatura],
            "chuva": [chuva],
            "fertilizante": [fertilizante]
        })

        resultado = modelo.predict(entrada)[0]

        st.success(f"💧 Irrigação recomendada: {resultado:.2f} litros")

        # Recomendação inteligente
        if resultado > 180:
            st.warning("🔴 Necessidade ALTA de irrigação")
        elif resultado > 130:
            st.info("🟡 Necessidade MODERADA de irrigação")
        else:
            st.success("🟢 Necessidade BAIXA de irrigação")

    st.markdown("---")
    st.markdown("✅ Sistema finalizado com sucesso!")


# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    main()