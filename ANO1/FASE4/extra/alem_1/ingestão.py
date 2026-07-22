# ingestion.py

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

SENSORES = json.loads(os.getenv("SENSORES", "[]"))
INTERVALO_COLETA_SEGUNDOS = int(os.getenv("INTERVALO_COLETA_SEGUNDOS", "5"))
from sensor import coletar_dados_sensor
from base_dados import inserir_leitura, listar_leituras


def executar_ingestao():
    """
    Executa continuamente a ingestão dos dados simulados IoT.
    """
    print("Iniciando ingestão automática de dados IoT...")
    print("Pressione CTRL + C para encerrar.\n")

    try:
        while True:
            for sensor in SENSORES:
                leitura = coletar_dados_sensor(sensor)

                inserir_leitura(
                    leitura["sensor_id"],
                    leitura["valor"],
                    leitura["data_hora"]
                )

                print(
                    f"Dado inserido: "
                    f"{leitura['nome']} | "
                    f"Valor: {leitura['valor']} {leitura['unidade']} | "
                    f"Data/Hora: {leitura['data_hora']}"
                )

            print("\nÚltimas leituras salvas no banco:")

            leituras = listar_leituras()

            for item in leituras:
                print(item)

            print("-" * 70)

            time.sleep(INTERVALO_COLETA_SEGUNDOS)

    except KeyboardInterrupt:
        print("\nIngestão encerrada pelo usuário.")