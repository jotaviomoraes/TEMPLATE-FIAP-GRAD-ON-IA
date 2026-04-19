# dados simulados
vetor_cultura <- c("morango", "banana", "morango", "banana")
vetor_area <- c(150,300,200,250)
vetor_produto <- c("herbicida", "herbicida", "NPK", "NPK")
vetor_litro_total <- c(30,60,100,125)

# ---------------------------------------
dados_plantio <- data.frame(
  Cultura = vetor_cultura,
  Area = vetor_area,
  Produto = vetor_produto,
  Litros_Totais = vetor_litro_total
)

print("Tabela de Dados Cadastrados")
print(dados_plantio)

# -------------------------
media_area <- mean(dados_plantio$Area)
desvio_area <- sd(dados_plantio$Area)
media_litros <- mean(dados_plantio$Litros_Totais)
desvio_litros <- sd(dados_plantio$Litros_Totais)


# ----------------------------------------------
cat("Resultados Estatísticos")
cat("Média da Área de plantio:", round(media_area, 2), "m²\n")
cat("Desvio Padrão da Área:", round(desvio_area, 2), "m²\n")
cat("-----------------------------------\n")
cat("Média de Litros aplicados:", round(media_litros, 2), "L\n")
cat("Desvio Padrão de Litros:", round(desvio_litros, 2), "L\n")


