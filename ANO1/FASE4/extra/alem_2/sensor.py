# sensor.py

import pandas as pd


def coletar_dados_sensor():
    """
    Simula a coleta de dados de sensores agrícolas.
    """

    novo_dado = pd.DataFrame([
        {
            "umidade": 55,
            "temperatura": 32,
            "chuva": 12,
            "fertilizante": 180
        }
    ])

    print("\nDados simulados do sensor:")
    print(novo_dado)

    return novo_dado