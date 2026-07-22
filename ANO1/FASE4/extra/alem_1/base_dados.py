import os
from pathlib import Path

import oracledb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

USUARIO = os.getenv("USUARIO")
SENHA = os.getenv("SENHA")
HOST = os.getenv("HOST")
PORTA = int(os.getenv("PORTA", "1521"))
SID = os.getenv("SID")


def conectar():
    dsn = oracledb.makedsn(
        host=HOST,
        port=PORTA,
        sid=SID
    )

    conexao = oracledb.connect(
        user=USUARIO,
        password=SENHA,
        dsn=dsn
    )

    return conexao


def testar_conexao():
    try:
        conexao = conectar()
        print("Conexão com Oracle FIAP realizada com sucesso!")
        conexao.close()
        return True

    except Exception as erro:
        print("Erro ao conectar no Oracle:")
        print(erro)
        return False


def inserir_dado(dado):
    conexao = conectar()
    cursor = conexao.cursor()

    sql = """
        INSERT INTO DADOS_METEOROLOGICOS (
            DATA_MEDICAO,
            HORA_MEDICAO,
            TEMPERATURA,
            UMIDADE,
            PRESSAO,
            PRECIPITACAO,
            VENTO
        ) VALUES (
            :data_medicao,
            :hora_medicao,
            :temperatura,
            :umidade,
            :pressao,
            :precipitacao,
            :vento
        )
    """

    cursor.execute(sql, dado)

    conexao.commit()
    cursor.close()
    conexao.close()


def listar_ultimos_registros(limite=5):
    conexao = conectar()
    cursor = conexao.cursor()

    sql = """
        SELECT
            DATA_MEDICAO,
            HORA_MEDICAO,
            TEMPERATURA,
            UMIDADE,
            PRESSAO,
            PRECIPITACAO,
            VENTO
        FROM DADOS_METEOROLOGICOS
        ORDER BY DATA_MEDICAO DESC, HORA_MEDICAO DESC
        FETCH FIRST :limite ROWS ONLY
    """

    cursor.execute(sql, {"limite": limite})
    registros = cursor.fetchall()

    cursor.close()
    conexao.close()

    return registros