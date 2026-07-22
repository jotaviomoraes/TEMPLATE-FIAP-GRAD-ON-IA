# modelo.py

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from configuracao import (
    VARIAVEIS_ENTRADA,
    ALVO,
    TAMANHO_TESTE,
    RANDOM_STATE,
    CAMINHO_MODELO
)


def treinar_modelo(dados):
    """
    Treina um modelo de regressão linear para prever irrigação.
    """

    x = dados[VARIAVEIS_ENTRADA]
    y = dados[ALVO]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=TAMANHO_TESTE,
        random_state=RANDOM_STATE
    )

    modelo = LinearRegression()

    modelo.fit(x_treino, y_treino)

    print("\nModelo treinado com sucesso!")

    return modelo, x_teste, y_teste


def salvar_modelo(modelo):
    """
    Salva o modelo treinado em arquivo.
    """

    os.makedirs("resultados/modelos", exist_ok=True)

    with open(CAMINHO_MODELO, "wb") as arquivo:
        pickle.dump(modelo, arquivo)

    print(f"Modelo salvo em: {CAMINHO_MODELO}")


def carregar_modelo_salvo():
    """
    Carrega um modelo salvo anteriormente.
    """

    with open(CAMINHO_MODELO, "rb") as arquivo:
        modelo = pickle.load(arquivo)

    print("Modelo carregado com sucesso!")

    return modelo