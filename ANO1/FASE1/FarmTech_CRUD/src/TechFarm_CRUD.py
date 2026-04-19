vetor_cultura = []
vetor_area = []
vetor_produto = []
vetor_litro_total = []

def exibir_menu():
    print("------------------------------------------------------------")
    print ("SISTEMA DE MENU")
    print ("1. Cadastrar item")
    print("2.mostrar cadastro")
    print("3 Atualizar")
    print("4 Remover dados")
    print("5 Sair")
    print("------------------------------------------------------------")


while True:
    exibir_menu()
    escolha = input("escolha uma opcao do menu")

    #---------------------------------------------------------------------
    if escolha == "1":
        print("Cadastro de dados")
        print("1.morango")
        print("2.banana")

        fruta = input("Digite o numero correspondente da fruta")
        print("------------------------------------------------------------")
        if fruta == "1":
            print("Cultivo de morango")
            fruta = "morango"

        elif fruta == "2" :
            print(" Cultivo de banana")
            fruta = "banana"
        else :
            print("nao existe")
            continue

        comprimento = float(input("Digite o comprimento do plantio"))
        largura = float(input("Digite o largura do plantio"))
        area = comprimento * largura
        produto = input("Digite o nome do produto a ser aplicado")
        quantidade = float(input(f"Digite a quantidade de litros de {produto} que será aplicado por metro quadrado"))
        total_produto = quantidade * area

        print("------------------------------------------------------------")
        print(f"para a área de {area} voce usará o total de {total_produto} litros de {produto}")
        print("------------------------------------------------------------")

        vetor_cultura.append(fruta)
        vetor_area.append(area)
        vetor_produto.append(produto)
        vetor_litro_total.append(total_produto)


#-----------------------------------------------------------------------
    elif escolha == "2":
        print(f"mostrando cadastro completo")
        print("------------------------------------------------------------")
        for i in range(len(vetor_cultura)):
            print(
                f"[{i}] cultura de: {vetor_cultura[i]} / Área: {vetor_area[i]} / Produto: {vetor_produto[i]} / quantidade total {vetor_litro_total[i]}")
        print("------------------------------------------------------------")



#---------------------------------------------------------------------
    elif escolha == "3":
        print("atualização de dados")

        for i in range(len(vetor_cultura)):
            print( f"Pressione [{i}] para alterar a cultura de: {vetor_cultura[i]} / Área: {vetor_area[i]} / Produto: {vetor_produto[i]}")


        indice = int(input("Digite o número do 'Índice' que deseja atualizar: "))

        if 0 <= indice < len(vetor_cultura):

            print(f"Atualizando registro [{indice}] - Cultura: {vetor_cultura[indice]}")

            novo_produto = input("Qual o novo produto a ser aplicado? ")
            novo_l = float(input(f"Digite a quantidade de litros de {novo_produto} que será aplicado por metro quadrado"))
            novo_total_produto = novo_l * vetor_area[indice]

            vetor_produto[indice] = novo_produto
            vetor_litro_total[indice] = novo_total_produto

            print("Registro atualizado com sucesso!")
            print("------------------------------------------------------------")
        else:
            print("Índice não encontrado.")
            print("------------------------------------------------------------")


#---------------------------------------------------------------------

    elif escolha == "4":
        print("deletar dados")

        for i in range(len(vetor_cultura)):
            print( f"Pressione [{i}] para deletar a cultura de: {vetor_cultura[i]} / Área: {vetor_area[i]} / Produto: {vetor_produto[i]}")

        indice = int(input("Digite o número do 'Índice' que deseja deletar: "))

        if 0 <= indice < len(vetor_cultura):

            print(f"deletando o registro [{indice}] - Cultura: {vetor_cultura[indice]}")

            vetor_cultura.pop(indice)
            vetor_area.pop(indice)
            vetor_produto.pop(indice)
            vetor_litro_total.pop(indice)

            print("Deleção realizada com sucesso!")

            print("------------------------------------------------------------")

        else:
            print("Índice não encontrado.")
            print("------------------------------------------------------------")

    elif escolha == "5":
        break

    else:
        print("opção invalida")
