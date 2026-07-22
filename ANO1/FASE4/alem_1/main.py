import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

INTERVALO_SEGUNDOS = int(os.getenv("INTERVALO_SEGUNDOS", "5"))
from sensor import gerar_dado_sensor
from base_dados import testar_conexao, inserir_dado, listar_ultimos_registros


def exibir_dado_inserido(dado):
    print("-" * 70)
    print("Novo dado IoT inserido no Oracle FIAP:")
    print(f"Data: {dado['data_medicao']}")
    print(f"Hora: {dado['hora_medicao']}")
    print(f"Temperatura: {dado['temperatura']} °C")
    print(f"Umidade: {dado['umidade']} %")
    print(f"Pressão: {dado['pressao']} hPa")
    print(f"Precipitação: {dado['precipitacao']} mm")
    print(f"Vento: {dado['vento']} m/s")


def exibir_ultimos_registros():
    print("\nÚltimos registros no banco Oracle:")

    registros = listar_ultimos_registros(5)

    for registro in registros:
        print(registro)


def main():
    print("Sistema de ingestão automática de dados IoT")
    print("Banco utilizado: Oracle FIAP")
    print("Tabela utilizada: DADOS_METEOROLOGICOS")

    conexao_ok = testar_conexao()

    if not conexao_ok:
        print("\nPrograma encerrado porque não foi possível conectar ao Oracle.")
        print("Verifique usuário, senha, host, porta e SID no arquivo .env.")
        sys.exit()

    print(f"\nInserindo novos dados a cada {INTERVALO_SEGUNDOS} segundos.")
    print("Pressione CTRL + C para encerrar.\n")

    try:
        while True:
            dado = gerar_dado_sensor()
            inserir_dado(dado)

            exibir_dado_inserido(dado)
            exibir_ultimos_registros()

            time.sleep(INTERVALO_SEGUNDOS)

    except KeyboardInterrupt:
        print("\nSistema encerrado pelo usuário.")


if __name__ == "__main__":
    main()