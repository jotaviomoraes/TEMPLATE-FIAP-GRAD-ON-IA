# visualizacao.py

import os
import matplotlib

# Usa backend que não precisa abrir janela gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from configuracao import CAMINHO_GRAFICO


def gerar_grafico(y_teste, previsoes):
    """
    Gera gráfico comparando valores reais e valores previstos.
    O gráfico é salvo em arquivo PNG, sem abrir janela na tela.
    """

    os.makedirs("resultados/graficos", exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(y_teste, previsoes)

    plt.xlabel("Valores reais de irrigação")
    plt.ylabel("Valores previstos de irrigação")
    plt.title("Comparação entre valores reais e previstos")

    plt.grid(True)

def gerar_grafico(y_teste, previsoes):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_teste, previsoes)
    ax.set_xlabel("Valores reais de irrigação")
    ax.set_ylabel("Valores previstos de irrigação")
    ax.set_title("Real vs Previsto")
    ax.grid(True)

    return fig


    print(f"\nGráfico salvo em: {CAMINHO_GRAFICO}")