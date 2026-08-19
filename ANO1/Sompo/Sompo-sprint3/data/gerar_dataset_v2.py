"""
gerar_dataset_v2.py
-------------------
Geracao do dataset simulado v2 do Sompo AgroPredict (Sprint 2).

Diferencas em relacao a v1 (Sprint 1):
- Volume: 1000 linhas (anteriormente ~50)
- Features ampliadas de 5 para 8:
    Originais: umidade_solo, declividade, chuva_24h, velocidade, status_operacao
    Novas (Sprint 2): temperatura, idade_maquina, horas_operacao

A logica de geracao mantem coerencia com o problema real:
- Velocidade alta + declividade alta -> risco por tombamento
- Velocidade alta + umidade alta    -> risco por perda de tracao
- Velocidade baixa + umidade alta + chuva recente -> risco de atolamento
- Temperatura alta + horas_operacao altas -> risco por fadiga / superaquecimento
- Idade da maquina alta + horas_operacao altas -> risco mecanico

Uso: python gerar_dataset_v2.py
Saida: dataset_v2.csv (mesma pasta)
"""

import numpy as np
import pandas as pd

SEED = 42
N_LINHAS = 1000

# Distribuicao estratificada por cenario: simula uma realidade onde
# existem operacoes seguras, situacoes-limite e operacoes criticas.
# Isso garante que o modelo veja exemplos suficientes de cada classe.
#
# Sprint 2 (revisao): adicionado cenario "intermediario" dedicado para
# reforcar a classe MEDIO, que estava sub-representada (~10% -> ~25%).
# A classe MEDIO e a mais dificil por ser fronteirica entre BAIXO e ALTO.
PROPORCAO_CENARIOS = {
    "seguro":        0.35,  # condicoes normais     -> tendencia a BAIXO
    "intermediario": 0.25,  # risco moderado         -> tendencia a MEDIO
    "limite":        0.22,  # situacoes-limite       -> tendencia a MEDIO/ALTO
    "critico":       0.18,  # cenarios adversos      -> tendencia a ALTO
}

np.random.seed(SEED)


def _amostra_cenario(cenario: str) -> dict:
    """
    Gera uma linha com distribuicoes deslocadas conforme o cenario.
    Distribuicoes calibradas para os limiares de alerta definidos:
      umidade>30, declividade>18, chuva>10, temperatura>38,
      velocidade>10, idade>10, horas>8
    """
    if cenario == "seguro":
        # Alvo: 0-2 alertas -> BAIXO
        # Variaveis bem abaixo dos limiares
        return {
            "umidade_solo":   np.clip(np.random.normal(20, 7),  5,  45),
            "declividade":    np.clip(np.random.normal(8,  4),  0,  20),
            "chuva_24h":      np.clip(np.random.normal(4,  3),  0,  15),
            "temperatura":    np.clip(np.random.normal(26, 5), 15,  38),
            "velocidade":     np.clip(np.random.normal(7,  2),  0,  12),
            "idade_maquina":  np.clip(np.random.normal(5,  3),  0,  12),
            "horas_operacao": np.clip(np.random.normal(5,  2),  0,   9),
        }
    if cenario == "intermediario":
        # Alvo: 3-5 alertas -> MEDIO
        # Metade das variaveis acima do limiar
        return {
            "umidade_solo":   np.clip(np.random.normal(35, 8),  5,  50),
            "declividade":    np.clip(np.random.normal(20, 4),  0,  28),
            "chuva_24h":      np.clip(np.random.normal(8,  5),  0,  30),
            "temperatura":    np.clip(np.random.normal(36, 4), 15,  43),
            "velocidade":     np.clip(np.random.normal(12, 3),  0,  22),
            "idade_maquina":  np.clip(np.random.normal(8,  3),  0,  15),
            "horas_operacao": np.clip(np.random.normal(9,  2),  0,  13),
        }
    if cenario == "limite":
        # Alvo: 4-6 alertas -> MEDIO/ALTO
        # Maioria acima do limiar, alguns proximos do MAX
        return {
            "umidade_solo":   np.clip(np.random.normal(40, 6),  5,  55),
            "declividade":    np.clip(np.random.normal(22, 3),  0,  30),
            "chuva_24h":      np.clip(np.random.normal(18, 8),  0,  45),
            "temperatura":    np.clip(np.random.normal(40, 3), 15,  47),
            "velocidade":     np.clip(np.random.normal(15, 3),  0,  25),
            "idade_maquina":  np.clip(np.random.normal(12, 3),  0,  20),
            "horas_operacao": np.clip(np.random.normal(10, 2),  0,  14),
        }
    # critico: alvo 6-7 alertas + valores MAX -> ALTO
    return {
        "umidade_solo":   np.clip(np.random.normal(50, 5), 30,  100),
        "declividade":    np.clip(np.random.normal(27, 3), 15,   30),
        "chuva_24h":      np.clip(np.random.normal(45, 10), 20, 100),
        "temperatura":    np.clip(np.random.normal(43, 3), 30,   45),
        "velocidade":     np.clip(np.random.normal(22, 4), 10,   30),
        "idade_maquina":  np.clip(np.random.normal(14, 3),  8,   20),
        "horas_operacao": np.clip(np.random.normal(12, 2),  8,   14),
    }


def gerar_features(n: int) -> pd.DataFrame:
    """Gera n linhas estratificadas pelos cenarios."""
    cenarios = np.random.choice(
        list(PROPORCAO_CENARIOS.keys()),
        size=n,
        p=list(PROPORCAO_CENARIOS.values())
    )
    linhas = [_amostra_cenario(c) for c in cenarios]
    df = pd.DataFrame(linhas)

    # Status de operacao tem correlacao com o cenario
    # Manobras concentram-se em situacoes de risco maior (terreno dificil)
    mapa_status = {
        "seguro":        [0.5, 0.3, 0.2],
        "intermediario": [0.45, 0.3, 0.25],
        "limite":        [0.45, 0.25, 0.3],
        "critico":       [0.4, 0.2, 0.4],
    }
    df["status_operacao"] = [
        np.random.choice(["Colheita", "Deslocamento", "Manobra"], p=mapa_status[c])
        for c in cenarios
    ]
    return df


def calcular_score(row) -> float:
    """
    Calcula um score de risco continuo (0-100) baseado em contagem de alertas.

    Limiar de ALERTA por variavel (acima disso = 1 alerta):
      umidade_solo   > 30%
      declividade    > 18%
      chuva_24h      > 10 mm
      temperatura    > 38 C
      velocidade     > 10 km/h
      idade_maquina  > 10 anos
      horas_operacao > 8 h

    Limiar MAXIMO (qualquer 1 ja classifica ALTO, mesmo sozinho):
      umidade_solo   >= 45%
      declividade    >= 25%
      chuva_24h      >= 40 mm
      temperatura    >= 45 C
      velocidade     >= 20 km/h

    Regras de classificacao:
      - Qualquer variavel em MAX   -> ALTO  (score >= 50)
      - 6 ou 7 alertas             -> ALTO  (score >= 50)
      - 3 a 5 alertas              -> MEDIO (score 25-49)
      - 0 a 2 alertas              -> BAIXO (score < 25)
    """

    # --- Conta quantas variaveis estao em alerta ---
    alertas = 0
    if row["umidade_solo"]   > 30: alertas += 1
    if row["declividade"]    > 18: alertas += 1
    if row["chuva_24h"]      > 10: alertas += 1
    if row["temperatura"]    > 38: alertas += 1
    if row["velocidade"]     > 10: alertas += 1
    if row["idade_maquina"]  > 10: alertas += 1
    if row["horas_operacao"] >  8: alertas += 1

    # --- Verifica se alguma variavel atingiu o limiar MAXIMO ---
    em_max = (
        row["umidade_solo"]  >= 45 or
        row["declividade"]   >= 25 or
        row["chuva_24h"]     >= 40 or
        row["temperatura"]   >= 45 or
        row["velocidade"]    >= 20
    )

    # --- Calcula score por mapa fixo (evita que ruido cruze fronteiras) ---
    # BAIXO: 0-2 alertas  -> score < 25
    # MEDIO: 3-5 alertas  -> score 25-49
    # ALTO:  6-7 alertas  -> score >= 50
    score_map = {0: 4, 1: 10, 2: 18, 3: 30, 4: 37, 5: 44, 6: 55, 7: 65}

    if em_max:
        score = 62 + alertas * 2   # garantido ALTO, mais alertas = score maior
    else:
        score = score_map[alertas]

    # Ruido muito pequeno para nao cruzar fronteiras de classe
    score += np.random.normal(0, 2)

    return float(np.clip(score, 0, 100))


def classificar(score: float) -> str:
    """Discretiza o score continuo em 3 classes de risco."""
    if score < 25:
        return "BAIXO"
    elif score < 50:
        return "MEDIO"
    else:
        return "ALTO"


def gerar_recomendacao(row, classe: str) -> str:
    """Gera recomendacao textual contextual baseada na causa dominante."""
    if classe == "BAIXO":
        return "Seguir rota atual"

    # Identifica a causa principal do risco para gerar uma recomendacao especifica
    if row["umidade_solo"] > 70 and row["chuva_24h"] > 20:
        return "Risco de atolamento - evitar areas saturadas"
    if row["velocidade"] > 15 and row["declividade"] > 15:
        return "Atencao a inclinacao - reduzir velocidade"
    if row["temperatura"] > 35 and row["horas_operacao"] > 10:
        return "Pausa recomendada - risco de fadiga"
    if row["idade_maquina"] > 12 and row["horas_operacao"] > 9:
        return "Verificar equipamento - desgaste elevado"
    if classe == "ALTO":
        return "Interromper operacao"
    return "Reduzir velocidade"


def main():
    print(f"Gerando {N_LINHAS} linhas com seed={SEED}...")
    df = gerar_features(N_LINHAS)

    # Calcula score, classe e recomendacao
    df["score_risco"]  = df.apply(calcular_score, axis=1).round(2)
    df["risco"]        = df["score_risco"].apply(classificar)
    df["recomendacao"] = df.apply(lambda r: gerar_recomendacao(r, r["risco"]), axis=1)

    # Reordena colunas
    colunas = [
        "umidade_solo", "declividade", "chuva_24h", "temperatura",
        "velocidade", "idade_maquina", "horas_operacao", "status_operacao",
        "score_risco", "risco", "recomendacao"
    ]
    df = df[colunas]

    # Arredonda numericas para legibilidade
    for col in ["umidade_solo", "declividade", "chuva_24h", "temperatura",
                "velocidade", "idade_maquina", "horas_operacao"]:
        df[col] = df[col].round(2)

    df.to_csv("dataset_v2.csv", index=False, encoding="utf-8")

    print("\n=== Distribuicao de classes ===")
    print(df["risco"].value_counts())
    print(f"\nProporcao:")
    print((df["risco"].value_counts(normalize=True) * 100).round(1))

    print("\n=== Primeiras 5 linhas ===")
    print(df.head())

    print(f"\nArquivo salvo: dataset_v2.csv ({len(df)} linhas, {len(df.columns)} colunas)")


if __name__ == "__main__":
    main()
