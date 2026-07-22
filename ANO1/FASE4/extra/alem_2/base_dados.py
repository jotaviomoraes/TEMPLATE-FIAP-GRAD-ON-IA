# base_dados.py

import pandas as pd
import os
from configuracao import CAMINHO_DADOS


def criar_base_dados():
    """
    Cria uma base de dados agrícola fictícia para teste do projeto.
    """

    dados = {
        "umidade": [
            30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
            32, 38, 43, 48, 53, 58, 63, 68, 72, 78
        ],
        "temperatura": [
            25, 26, 27, 28, 30, 31, 32, 33, 34, 35,
            24, 27, 29, 30, 31, 33, 34, 35, 36, 37
        ],
        "chuva": [
            5, 8, 10, 12, 15, 18, 20, 22, 25, 28,
            4, 7, 9, 11, 14, 17, 19, 21, 24, 27
        ],
        "fertilizante": [
            100, 120, 140, 150, 170, 190, 210, 230, 250, 270,
            110, 130, 145, 160, 180, 200, 220, 240, 260, 280
        ],
        "irrigacao": [
            220, 210, 200, 185, 175, 160, 150, 135, 120, 110,
            225, 205, 195, 180, 165, 155, 140, 130, 115, 100
        ],
        "rendimento": [
            900, 980, 1050, 1130, 1200, 1280, 1350, 1420, 1500, 1580,
            920, 1000, 1080, 1160, 1240, 1320, 1390, 1460, 1540, 1620
        ]
    }

    tabela = pd.DataFrame(dados)

    os.makedirs("dados", exist_ok=True)

    tabela.to_csv(CAMINHO_DADOS, index=False, encoding="utf-8")

    print("Base de dados criada com sucesso!")
    print(f"Arquivo salvo em: {CAMINHO_DADOS}")

    return tabela