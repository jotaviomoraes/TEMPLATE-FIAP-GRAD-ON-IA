# JoaoOtavioMoraes_rm573227_fase2_cap7

# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS (Simulando os dados da tabela fornecida)
# ==============================================================================

# Variável Quantitativa: Produtividade Média (kg/ha)
produtividade <- c(3758, 3462, 3280, 3592, 3415, 3810, 3540, 3430, 3350, 6015, 
                   5940, 5820, 5650, 5710, 5560, 5420, 7850, 5210, 2650, 4120, 
                   2430, 1650, 1530, 1710, 1120, 3290, 2940, 3450, 4612, 4390)

# Variável Qualitativa: Nível de Desempenho da Safra
desempenho <- c("Regular", "Regular", "Fraco", "Regular", "Regular", "Bom", 
                "Regular", "Regular", "Regular", "Bom", "Bom", "Bom", "Regular", 
                "Regular", "Fraco", "Regular", "Regular", "Regular", "Fraco", 
                "Regular", "Fraco", "Bom", "Regular", "Bom", "Fraco", "Bom", 
                "Regular", "Bom", "Ótimo", "Ótimo")

# Organizando os níveis (fator ordinal) para o gráfico qualitativo fazer sentido
desempenho <- factor(desempenho, levels = c("Fraco", "Regular", "Bom", "Ótimo"), ordered = TRUE)


# ==============================================================================
# 2. ANÁLISE DA VARIÁVEL QUANTITATIVA (Produtividade Média)
# ==============================================================================

cat("--- MEDIDAS DE TENDÊNCIA CENTRAL ---\n")

# Média
media_prod <- mean(produtividade)
cat("Média:", round(media_prod, 2), "kg/ha\n")

# Mediana
mediana_prod <- median(produtividade)
cat("Mediana:", mediana_prod, "kg/ha\n")


cat("--- MEDIDAS DE DISPERSÃO ---\n")

# Variância
var_prod <- var(produtividade)
cat("Variância:", round(var_prod, 2), "\n")

# Desvio Padrão
dp_prod <- sd(produtividade)
cat("Desvio Padrão:", round(dp_prod, 2), "kg/ha\n")

# Amplitude Total
amplitude_prod <- max(produtividade) - min(produtividade)
cat("Amplitude Total:", amplitude_prod, "kg/ha\n")

# Intervalo Interquartílico (IQR)
iqr_prod <- IQR(produtividade)
cat("Intervalo Interquartílico (IQR):", iqr_prod, "kg/ha\n\n")


cat("--- MEDIDAS SEPARATRIZES ---\n")

# Quartis (25%, 50%, 75%)
quartis <- quantile(produtividade, probs = c(0.25, 0.50, 0.75))
cat("Quartis:\n")
print(quartis)

# Decis (10%, 20%, ..., 90%)
decis <- quantile(produtividade, probs = seq(0.1, 0.9, by = 0.1))
cat("\nDecis:\n")
print(decis)


# ==============================================================================
# 3. ANÁLISE GRÁFICA - VARIÁVEL QUANTITATIVA
# ==============================================================================



# Histograma
hist(produtividade, 
     main = "Histograma da Produtividade", 
     xlab = "Produtividade (kg/ha)", 
     ylab = "Frequência", 
     col = "steelblue", 
     border = "white")


# ==============================================================================
# 4. ANÁLISE GRÁFICA - VARIÁVEL QUALITATIVA (Nível de Desempenho)
# ==============================================================================


# Tabela de frequência
tabela_freq <- table(desempenho)

# Gráfico de Barras
barplot(tabela_freq, 
        main = "Desempenho da Safra por Categoria", 
        xlab = "Nível de Desempenho", 
        ylab = "Quantidade de Ocorrências", 
        col = c("firebrick", "orange", "mediumseagreen", "forestgreen"), 
        ylim = c(0, max(tabela_freq) + 2))
        
        