# ingestao.py

def verificar_dados(dados):
    """
    Verifica informações básicas da base de dados.
    """

    print("\nInformações da base de dados:")
    print(dados.info())

    print("\nEstatísticas descritivas:")
    print(dados.describe())

    print("\nValores ausentes por coluna:")
    print(dados.isnull().sum())


def limpar_dados(dados):
    """
    Remove valores ausentes, caso existam.
    """

    dados_limpos = dados.dropna()

    print("\nDados limpos com sucesso!")

    return dados_limpos