"""
inserir_predicoes.py
--------------------
Script de ingestao das predicoes do modelo no Oracle XE.

Carrega o modelo treinado (modelo.pkl), processa um lote de predicoes
e persiste no banco. Tambem dispara o registro de alertas para predicoes
classificadas como ALTO.

Uso:
    python inserir_predicoes.py [--n LINHAS] [--modo {batch|exemplo}]

Pre-requisitos:
    - Oracle XE rodando localmente
    - Schema criado (executar schema.sql primeiro)
    - Dependencias: oracledb, joblib, pandas, scikit-learn
    - Variaveis de ambiente (ou edicao do dicionario CONFIG abaixo):
        ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN
"""

import argparse
import os
import sys
from pathlib import Path
import random

import joblib
import oracledb
import pandas as pd

# =====================================================================
# Configuracao
# =====================================================================
CONFIG = {
    "user":     os.getenv("ORACLE_USER", "system"),
    "password": os.getenv("ORACLE_PASSWORD", "admin"),
    "dsn":      os.getenv("ORACLE_DSN", "localhost:1521/XE"),
}

# Caminhos relativos ao script
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "modelo" / "modelo.pkl"
DATASET_PATH = BASE_DIR / "data" / "dataset_v2.csv"

# Mapeamento de recomendacao por classe
RECOMENDACOES = {
    "BAIXO": "Seguir rota atual",
    "MEDIO": "Reduzir velocidade",
    "ALTO":  "Interromper operacao",
}


# =====================================================================
# Funcoes auxiliares
# =====================================================================
def conectar():
    """Estabelece conexao com o Oracle XE."""
    try:
        conn = oracledb.connect(**CONFIG)
        print(f"[OK] Conectado ao Oracle: {CONFIG['dsn']}")
        return conn
    except oracledb.DatabaseError as e:
        print(f"[ERRO] Falha na conexao: {e}")
        print("Verifique se o Oracle XE esta rodando e o schema foi criado.")
        sys.exit(1)


def carregar_modelo():
    """Carrega o modelo treinado a partir do .pkl."""
    if not MODELO_PATH.exists():
        print(f"[ERRO] Modelo nao encontrado em {MODELO_PATH}")
        print("Execute o notebook modelo/random_forest.ipynb primeiro.")
        sys.exit(1)
    modelo = joblib.load(MODELO_PATH)
    print(f"[OK] Modelo carregado: {MODELO_PATH.name}")
    return modelo


def gerar_recomendacao_contextual(row: pd.Series, classe: str) -> str:
    """Gera recomendacao contextual baseada na causa dominante do risco."""
    if classe == "BAIXO":
        return RECOMENDACOES["BAIXO"]
    if row["umidade_solo"] > 70 and row["chuva_24h"] > 20:
        return "Risco de atolamento - evitar areas saturadas"
    if row["velocidade"] > 15 and row["declividade"] > 15:
        return "Atencao a inclinacao - reduzir velocidade"
    if row["temperatura"] > 35 and row["horas_operacao"] > 10:
        return "Pausa recomendada - risco de fadiga"
    if row["idade_maquina"] > 12 and row["horas_operacao"] > 9:
        return "Verificar equipamento - desgaste elevado"
    return RECOMENDACOES[classe]


def inserir_predicao(cur, row: pd.Series, classe: str, score: float,
                      probs: dict, recomendacao: str,
                      id_equipamento: int, id_operador: int) -> int:
    """Insere uma predicao na tabela. Retorna id_predicao gerado."""
    sql = """
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
        RETURNING id_predicao INTO :id_out
    """
    id_out = cur.var(oracledb.NUMBER)
    cur.execute(sql, {
        "equip":  id_equipamento,
        "oper":   id_operador,
        "umid":   float(row["umidade_solo"]),
        "decl":   float(row["declividade"]),
        "chuva":  float(row["chuva_24h"]),
        "temp":   float(row["temperatura"]),
        "vel":    float(row["velocidade"]),
        "idade":  float(row["idade_maquina"]),
        "horas":  float(row["horas_operacao"]),
        "status": str(row["status_operacao"]),
        "score":  float(score),
        "classe": classe,
        "reco":   recomendacao,
        "pb":     float(probs.get("BAIXO", 0)),
        "pm":     float(probs.get("MEDIO", 0)),
        "pa":     float(probs.get("ALTO", 0)),
        "versao": "v1.0",
        "id_out": id_out
    })
    return int(id_out.getvalue()[0])


def inserir_alerta(cur, id_predicao: int):
    """Registra um alerta para predicoes de risco ALTO."""
    cur.execute("""
        INSERT INTO alertas (id_alerta, id_predicao)
        VALUES (seq_alerta.NEXTVAL, :id_pred)
    """, {"id_pred": id_predicao})


def processar_lote(conn, modelo, df: pd.DataFrame):
    """Processa um DataFrame e insere todas as predicoes no banco."""
    features = ["umidade_solo", "declividade", "chuva_24h", "temperatura",
                "velocidade", "idade_maquina", "horas_operacao", "status_operacao"]
    X = df[features]

    classes_pred = modelo.predict(X)
    probas = modelo.predict_proba(X)
    classes_modelo = modelo.classes_

    cur = conn.cursor()
    qtd_inseridos = 0
    qtd_alertas = 0

    # Equipamentos e operadores existentes no banco
    ids_equip = [1, 2, 3]
    ids_oper  = [1, 2, 3]

    for i, (idx, row) in enumerate(df.iterrows()):
        classe = classes_pred[i]
        score = probas[i].max() * 100
        probs = {c: probas[i][j] for j, c in enumerate(classes_modelo)}
        recomendacao = gerar_recomendacao_contextual(row, classe)

        # Atribuicao pseudo-aleatoria de equipamento/operador
        id_eq  = random.choice(ids_equip)
        id_op  = random.choice(ids_oper)

        id_pred = inserir_predicao(
            cur, row, classe, score, probs, recomendacao, id_eq, id_op
        )
        qtd_inseridos += 1

        if classe == "ALTO":
            inserir_alerta(cur, id_pred)
            qtd_alertas += 1

    conn.commit()
    cur.close()
    print(f"[OK] {qtd_inseridos} predicoes inseridas")
    print(f"[OK] {qtd_alertas} alertas registrados (risco ALTO)")


def relatorio_apos_ingestao(conn):
    """Imprime um resumo do que foi persistido."""
    cur = conn.cursor()
    print("\n=== Resumo do banco apos ingestao ===")

    cur.execute("SELECT COUNT(*) FROM predicoes")
    print(f"  Total de predicoes: {cur.fetchone()[0]}")

    cur.execute("""
        SELECT classe_risco, COUNT(*)
        FROM predicoes
        GROUP BY classe_risco
        ORDER BY classe_risco
    """)
    print("  Distribuicao por classe:")
    for classe, qtd in cur.fetchall():
        print(f"    {classe:6}: {qtd}")

    cur.execute("SELECT COUNT(*) FROM alertas")
    print(f"  Alertas registrados: {cur.fetchone()[0]}")

    cur.execute("SELECT * FROM vw_metricas_diarias ORDER BY dia DESC FETCH FIRST 5 ROWS ONLY")
    print("\n  Metricas diarias (top 5):")
    for row in cur.fetchall():
        print(f"    {row}")

    cur.close()


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Ingestao de predicoes no Oracle XE")
    parser.add_argument("--n", type=int, default=50,
                        help="Numero de linhas do dataset a processar (default: 50)")
    parser.add_argument("--modo", choices=["batch", "exemplo"], default="batch",
                        help="batch: amostra do dataset | exemplo: 3 cenarios fixos")
    args = parser.parse_args()

    print("=" * 60)
    print("Sompo AgroPredict - Ingestao de predicoes no Oracle XE")
    print("=" * 60)

    modelo = carregar_modelo()
    conn = conectar()

    if args.modo == "batch":
        df = pd.read_csv(DATASET_PATH).sample(n=args.n, random_state=42)
        print(f"[INFO] Processando {len(df)} amostras do dataset...")
    else:
        # 3 cenarios fixos para validacao manual
        df = pd.DataFrame([
            {"umidade_solo": 35, "declividade": 5,  "chuva_24h": 2,
             "temperatura": 25, "velocidade": 12, "idade_maquina": 4,
             "horas_operacao": 5,  "status_operacao": "Deslocamento"},
            {"umidade_solo": 65, "declividade": 12, "chuva_24h": 15,
             "temperatura": 33, "velocidade": 16, "idade_maquina": 8,
             "horas_operacao": 9,  "status_operacao": "Colheita"},
            {"umidade_solo": 88, "declividade": 22, "chuva_24h": 50,
             "temperatura": 39, "velocidade": 22, "idade_maquina": 15,
             "horas_operacao": 12, "status_operacao": "Manobra"},
        ])
        print(f"[INFO] Processando 3 cenarios de exemplo...")

    processar_lote(conn, modelo, df)
    relatorio_apos_ingestao(conn)

    conn.close()
    print("\n[OK] Processo concluido.")


if __name__ == "__main__":
    main()
