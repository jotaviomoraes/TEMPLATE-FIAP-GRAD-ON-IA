"""
persistencia.py
----------------
Registra cada predicao do dashboard em DUAS camadas, sem que uma dependa
da outra:

1) SQLite LOCAL (`data/historico_local.db`) — uma tabela que cresce
   sozinha a cada leitura, funciona 100% offline e nao depende do Oracle
   estar no ar. Ela tambem serve como o embriao de um dataset REAL — algo
   que hoje o projeto ainda nao tem (ver explicacao no chat / README).

2) Oracle XE (mesmo schema de sql/schema.sql) — melhor esforco: se o
   banco nao estiver acessivel, a gravacao local continua funcionando e
   nada quebra na tela do operador.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_LOCAL_PATH = Path(__file__).resolve().parent / "data" / "historico_local.db"


# =====================================================================
# SQLite local
# =====================================================================
def _conectar_local() -> sqlite3.Connection:
    DB_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_LOCAL_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historico_predicoes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora        TEXT NOT NULL,
            umidade_solo     REAL,
            declividade      REAL,
            chuva_24h        REAL,
            temperatura      REAL,
            velocidade       REAL,
            idade_maquina    REAL,
            horas_operacao   REAL,
            status_operacao  TEXT,
            classe_risco     TEXT,
            score_risco      REAL,
            recomendacao     TEXT,
            fonte_velocidade TEXT,
            fonte_clima      TEXT,
            fonte_horas      TEXT,
            gravado_oracle   INTEGER DEFAULT 0
        )
    """)
    return conn


def registrar_local(registro: dict, gravado_oracle: bool) -> int:
    """Insere uma linha e retorna o id gerado."""
    conn = _conectar_local()
    cur = conn.execute("""
        INSERT INTO historico_predicoes (
            data_hora, umidade_solo, declividade, chuva_24h, temperatura,
            velocidade, idade_maquina, horas_operacao, status_operacao,
            classe_risco, score_risco, recomendacao,
            fonte_velocidade, fonte_clima, fonte_horas, gravado_oracle
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        registro.get("umidade_solo"), registro.get("declividade"),
        registro.get("chuva_24h"), registro.get("temperatura"),
        registro.get("velocidade"), registro.get("idade_maquina"),
        registro.get("horas_operacao"), registro.get("status_operacao"),
        registro.get("classe_risco"), registro.get("score_risco"),
        registro.get("recomendacao"),
        registro.get("fonte_velocidade"), registro.get("fonte_clima"),
        registro.get("fonte_horas"), int(gravado_oracle),
    ))
    conn.commit()
    novo_id = cur.lastrowid
    conn.close()
    return novo_id


def carregar_local(limite: int = 200):
    """Retorna um DataFrame-friendly (colunas, linhas) com as ultimas N leituras."""
    import pandas as pd
    conn = _conectar_local()
    df = pd.read_sql(
        "SELECT * FROM historico_predicoes ORDER BY id DESC LIMIT ?",
        conn, params=(limite,)
    )
    conn.close()
    return df


def contar_local() -> int:
    conn = _conectar_local()
    n = conn.execute("SELECT COUNT(*) FROM historico_predicoes").fetchone()[0]
    conn.close()
    return n


# =====================================================================
# Oracle (best-effort)
# =====================================================================
def registrar_oracle(config_oracle: dict, registro: dict,
                      id_equipamento: int = 1, id_operador: int = 1) -> bool:
    """
    Tenta gravar no Oracle usando o MESMO schema de sql/schema.sql.
    Retorna True se conseguiu, False se falhou -- NUNCA lanca excecao,
    para o dashboard continuar funcionando mesmo sem o banco no ar.
    """
    conn = None
    try:
        import oracledb
        conn = oracledb.connect(**config_oracle)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predicoes (
                id_predicao, id_equipamento, id_operador,
                umidade_solo, declividade, chuva_24h, temperatura,
                velocidade, idade_maquina, horas_operacao, status_operacao,
                score_risco, classe_risco, recomendacao,
                prob_baixo, prob_medio, prob_alto, versao_modelo
            ) VALUES (
                seq_predicao.NEXTVAL, :equip, :oper,
                :umid, :decl, :chuva, :temp,
                :vel, :idade, :horas, :status,
                :score, :classe, :reco,
                :pb, :pm, :pa, :versao
            )
        """, {
            "equip": id_equipamento, "oper": id_operador,
            "umid": float(registro["umidade_solo"]), "decl": float(registro["declividade"]),
            "chuva": float(registro["chuva_24h"]), "temp": float(registro["temperatura"]),
            "vel": float(registro["velocidade"]), "idade": float(registro["idade_maquina"]),
            "horas": float(registro["horas_operacao"]), "status": registro["status_operacao"],
            "score": float(registro["score_risco"]), "classe": registro["classe_risco"],
            "reco": registro["recomendacao"],
            "pb": float(registro.get("prob_baixo") or 0),
            "pm": float(registro.get("prob_medio") or 0),
            "pa": float(registro.get("prob_alto") or 0),
            "versao": "v1.0",
        })
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
