# preditivo.py

from sensor import coletar_dados_sensor


def fazer_previsao(modelo):
    """
    Faz previsão de irrigação usando dados simulados de sensores.
    """

    novo_dado = coletar_dados_sensor()

    previsao = modelo.predict(novo_dado)

    valor_previsto = previsao[0]

    print("\nResultado da previsão:")
    print(f"Irrigação recomendada: {valor_previsto:.2f} litros")

    if valor_previsto > 180:
        print("Recomendação: necessidade alta de irrigação.")
    elif valor_previsto > 130:
        print("Recomendação: necessidade moderada de irrigação.")
    else:
        print("Recomendação: necessidade baixa de irrigação.")

    return valor_previsto