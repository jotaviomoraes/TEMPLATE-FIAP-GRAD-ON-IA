# configuracao.py

CAMINHO_DADOS = "dados/dados_agricolas.csv"

CAMINHO_MODELO = "resultados/modelos/modelo_irrigacao.pkl"

CAMINHO_GRAFICO = "resultados/graficos/grafico_real_vs_previsto.png"

VARIAVEIS_ENTRADA = [
    "umidade",
    "temperatura",
    "chuva",
    "fertilizante"
]

ALVO = "irrigacao"

TAMANHO_TESTE = 0.2

RANDOM_STATE = 42