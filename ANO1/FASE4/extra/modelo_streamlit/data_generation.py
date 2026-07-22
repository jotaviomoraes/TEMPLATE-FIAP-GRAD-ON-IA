import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=True)  ## usando cache
def generate_data(n_samples=1000):
    """
    Gera um dataset simulado de produção agrícola.

    Args:
        n_samples (int): Número de amostras a serem geradas.

    Returns:
        pd.DataFrame: DataFrame contendo o dataset simulado.
    """
    np.random.seed(42)

    # Gerando variáveis climáticas com distribuições realistas
    temperatura = np.random.normal(loc=22, scale=3, size=n_samples)  # Temperatura média de 22°C
    precipitacao = np.random.gamma(shape=2, scale=30, size=n_samples)  # Precipitação em mm
    umidade = np.random.uniform(low=50, high=80, size=n_samples)  # Umidade relativa em %

    # Variáveis categóricas
    fertilizante = np.random.choice(['Orgânico', 'Sintético'], size=n_samples, p=[0.4, 0.6])
    tipo_solo = np.random.choice(['Arenoso', 'Argiloso', 'Siltoso'], size=n_samples, p=[0.3, 0.5, 0.2])

    # Mapear categorias para valores numéricos (efeitos)
    mapa_fertilizante = {'Orgânico': 1.05, 'Sintético': 1.10}
    mapa_solo = {'Arenoso': 0.95, 'Argiloso': 1.00, 'Siltoso': 1.02}

    efeito_fertilizante = np.vectorize(mapa_fertilizante.get)(fertilizante)
    efeito_solo = np.vectorize(mapa_solo.get)(tipo_solo)

    # Calculando a produção com interações mais realistas
    producao = (
            (temperatura - 20) * 2 +  # Aumento de produção por temperatura acima de 20°C
            (precipitacao - 50) * 0.5 +  # Aumento por precipitação acima de 50mm
            (umidade - 50) * 0.3  # Aumento por umidade acima de 50%
               ) * efeito_fertilizante * efeito_solo

    # Adicionando variabilidade aleatória
    producao += np.random.normal(loc=0, scale=5, size=n_samples)

    # Garantindo que a produção não seja negativa
    producao = np.clip(producao, a_min=0, a_max=None)

    # Criando o DataFrame final
    df = pd.DataFrame({
        'Temperatura': temperatura,
        'Precipitação': precipitacao,
        'Umidade': umidade,
        'Fertilizante': fertilizante,
        'Tipo de Solo': tipo_solo,
        'Produção': producao
    })

    return df