import json
import oracledb
import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
from dotenv import load_dotenv


# traz o .env para a memória
load_dotenv()



# ---------------------------DICIONARIO

def calcular_perda(toneladas, metodo):

    if metodo.lower() == 'manual':
        percentual = 0.05
    elif metodo.lower() == 'mecanico':
        percentual = 0.15
    else:
        percentual = 0.0

    perda_toneladas = toneladas * percentual

    # valor medio atual da cana.
    valor_tonelada_euro = 30.0
    prejuizo_estimado = perda_toneladas * valor_tonelada_euro

    return perda_toneladas, prejuizo_estimado


def cadastrar_talhao(lista_talhoes):

    print("\n--- Novo Cadastro de Talhão ---")
    try:
        id_talhao = int(input("Digite o ID numérico do Talhão (ex: 101): "))

        area = float(input("Área plantada (em hectares): "))
        if area <= 0:
            print("Erro: A área deve ser maior que zero!")
            return

        toneladas = float(input("Estimativa total de produção (em toneladas): "))

        print("Métodos de colheita: [1] Manual (5% perda) | [2] Mecânico (15% perda)")
        opcao_metodo = input("Escolha o método (1 ou 2): ")

        if opcao_metodo == '1':
            metodo = 'Manual'
        elif opcao_metodo == '2':
            metodo = 'Mecanico'
        else:
            metodo = 'Desconhecido'

        if metodo == 'Desconhecido':
            print("Erro: Método inválido. Cadastro cancelado.")
            return

        perda_ton, prejuizo_eur = calcular_perda(toneladas, metodo)


        talhao = {
            "id_talhao": id_talhao,
            "area_ha": area,
            "producao_estimada_ton": toneladas,
            "metodo_colheita": metodo,
            "perda_estimada_ton": round(perda_ton, 2),
            "prejuizo_estimado_eur": round(prejuizo_eur, 2)
        }

        lista_talhoes.append(talhao)
        print(f"\n[SUCESSO] Talhão {id_talhao} cadastrado e analisado!")

    except:

        print("\n[ERRO DE TIPO] Por favor, digite apenas números nos campos de ID, Área e Produção.")



# 2. ----------------------------------------------------JSON


def salvar_json(lista_talhoes):
    
    
    if not lista_talhoes:
        print("\nNão há dados em memória para salvar.")
        return

    try:
        with open('../scripts/relatorio_safra.json', 'w', encoding='utf-8') as arquivo:
            json.dump(lista_talhoes, arquivo, ensure_ascii=False, indent=4)
        print("\n[SUCESSO] Relatório exportado como 'relatorio_safra.json' no diretório atual.")
    except Exception as e:
        print(f"\n[ERRO] Falha ao gravar arquivo JSON: {e}")



# 3. ----------------------------------------- ORACLE


def exportar_para_oracle(lista_talhoes):

    if not lista_talhoes:
        print("\n[Aviso] Não há dados em memória para exportar ao banco.")
        return

    print("\n--- Iniciando conexão com Oracle Database ---")
    try:
        conexao = oracledb.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            dsn=os.getenv("DB_DSN")
        )


        cursor = conexao.cursor()
        print("[OK] Conexão estabelecida com sucesso!")


        inseridos = 0


        for t in lista_talhoes:
            sql = """
            INSERT INTO RM_TALHOES (ID_TALHAO, AREA_HA, PRODUCAO_TON, METODO, PERDA_TON, PREJUIZO_EUR)
            VALUES (:1, :2, :3, :4, :5, :6)
            """
            dados = (
                t['id_talhao'],
                t['area_ha'],
                t['producao_estimada_ton'],
                t['metodo_colheita'],
                t['perda_estimada_ton'],
                t['prejuizo_estimado_eur']
            )

            try:
                cursor.execute(sql, dados)
                print(f" -> Talhão ID {t['id_talhao']} inserido no banco.")
                inseridos += 1
            except oracledb.IntegrityError:
                print(f" -> [ERRO] O Talhão ID {t['id_talhao']} já existe no banco (ID Duplicado).")


        conexao.commit()


        cursor.close()
        conexao.close()

        print(f"\n[SUCESSO] Operação finalizada. {inseridos} registro(s) gravado(s) no Oracle!")

    except oracledb.DatabaseError as e:
        erro, = e.args
        print(f"\n[ERRO DE BANCO] Falha na comunicação: Código {erro.code} - {erro.message}")
    except Exception as e:
        print(f"\n[ERRO GERAL] Ocorreu um problema inesperado: {e}")



#  ------------------------------------------------- PROGRAMA


def main():

    banco_em_memoria = []

    # se ja existir um .json
    if os.path.exists('../scripts/relatorio_safra.json'):
        try:
            with open('../scripts/relatorio_safra.json', 'r', encoding='utf-8') as f:
                banco_em_memoria = json.load(f)
            print(f"\n[SISTEMA] {len(banco_em_memoria)} registros antigos carregados com sucesso!")
        except Exception:
            print("\n[AVISO] Arquivo JSON encontrado, mas está vazio ou corrompido.")

    # ----------------------------------------------------MAIN

    while True:
        print("\n" + "=" * 40)
        print("   OTIMIZADOR DE FAZENDA PARA CANA-DE-AÇÚCAR   ")
        print("=" * 40)
        print("1 - Registrar Talhão e Calcular Perdas")
        print("2 - Exibir Tabela de Memória Atual")
        print("3 - Exportar Dados para Arquivo JSON")
        print("4 - Sincronizar com Banco Oracle (DB)")
        print("0 - Encerrar Sistema")

        opcao = input("\nEscolha a operação desejada: ")

        if opcao == '1':
            cadastrar_talhao(banco_em_memoria)
        elif opcao == '2':
            if not banco_em_memoria:
                print("\n[Aviso] A tabela de memória está vazia.")
            else:
                print("\n--- Dados Atuais em Memória (Tabela Pandas) ---")
                df = pd.DataFrame(banco_em_memoria)
                print(df)
        elif opcao == '3':
            salvar_json(banco_em_memoria)
        elif opcao == '4':
            exportar_para_oracle(banco_em_memoria)
        elif opcao == '0':
            print("\nEncerrando o AgroHarvest Optimizer. Até logo!")
            break
        else:
            print("\n[ERRO] Opção não reconhecida. Tente novamente.")


if __name__ == "__main__":
    main()