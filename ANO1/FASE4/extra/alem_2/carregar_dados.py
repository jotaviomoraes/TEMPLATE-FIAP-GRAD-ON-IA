# carregar_dados.py

import pandas as pd
import os
from configuracao import CAMINHO_DADOS
from base_dados import criar_base_dados


def carregar_dados():
    """
    Carrega a base de dados.
    Caso o arquivo CSV não exista, cria uma base de exemplo.
    """

    if not os.path.exists(CAMINHO_DADOS):
        print("Base de dados não encontrada.")
        print("Criando base de dados de exemplo...")
        criar_base_dados()

    dados = pd.read_csv(CAMINHO_DADOS, encoding="utf-8")

    print("Dados carregados com sucesso!")
    print(dados.head())

    return dados