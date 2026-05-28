import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from data_generation import generate_data  # Carrega o dataset

# Configurações gerais
st.set_page_config(page_title='Análise de Dados', layout='wide')

# Título do aplicativo
st.title('Exploração de Dados')

# ================================================
# 1. Carregar os Dados
# ================================================
st.header('1. Carregar os Dados')

# Carregar os dados gerados
df = generate_data()

# Mostrar os primeiros registros
st.subheader('Visualização dos Dados')
st.dataframe(df.head())

# ================================================
# 2. Visão Geral dos Dados
# ================================================
st.header('2. Visão Geral dos Dados')

# Informações sobre o DataFrame
st.subheader('Informações do DataFrame')
st.write('Dimensões do DataFrame:')
st.write(f'Linhas: {df.shape[0]}, Colunas: {df.shape[1]}')

st.subheader('Tipos de Dados')
# Converter os tipos de dados para string
df_types = pd.DataFrame({
    'Coluna': df.columns,
    'Tipos de Dados': df.dtypes.astype(str)
})
st.write(df_types)

# Verificar valores ausentes
st.subheader('Valores Ausentes')
st.write(df.isnull().sum())

# Estatísticas Descritivas
st.subheader('Estatísticas Descritivas')
st.write(df.describe())

# ================================================
# 3. Análise Univariada
# ================================================
st.header('3. Análise Univariada')

numeric_columns = ['Temperatura', 'Precipitação', 'Umidade', 'Produção']
categorical_columns = ['Fertilizante', 'Tipo de Solo']

# Histogramas das variáveis numéricas
st.subheader('Distribuições das Variáveis Numéricas')
for col in numeric_columns:
    st.write(f'**{col}**')
    fig = px.histogram(df, x=col, nbins=30, title=f'Distribuição de {col}')
    st.plotly_chart(fig, use_container_width=True)

# Box plots das variáveis numéricas
st.subheader('Box Plots das Variáveis Numéricas')
for col in numeric_columns:
    st.write(f'**{col}**')
    fig = px.box(df, y=col, points='all', title=f'Box Plot de {col}')
    st.plotly_chart(fig, use_container_width=True)

# Distribuição das variáveis categóricas
st.subheader('Distribuições das Variáveis Categóricas')
for col in categorical_columns:
    st.write(f'**{col}**')
    fig = px.histogram(df, x=col, title=f'Distribuição de {col}')
    st.plotly_chart(fig, use_container_width=True)

# ================================================
# 4. Análise Bivariada
# ================================================
st.header('4. Análise Bivariada')

# Gráficos de dispersão entre variáveis numéricas
st.subheader('Gráficos de Dispersão entre Variáveis Numéricas')
variable_pairs = [
    ('Temperatura', 'Produção'),
    ('Precipitação', 'Produção'),
    ('Umidade', 'Produção'),
    ('Temperatura', 'Precipitação'),
    ('Temperatura', 'Umidade')
]

for x_var, y_var in variable_pairs:
    st.write(f'**{x_var} vs {y_var}**')
    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        color='Fertilizante',
        symbol='Tipo de Solo',
        title=f'{y_var} vs {x_var}'
    )
    st.plotly_chart(fig, use_container_width=True)

# Distribuição de Produção por Fertilizante e Tipo de Solo
st.subheader('Distribuição de Produção por Fertilizante e Tipo de Solo')

# Por Fertilizante
st.write('**Produção por Fertilizante**')
fig = px.violin(
    df,
    x='Fertilizante',
    y='Produção',
    box=True,
    points='all',
    title='Distribuição de Produção por Fertilizante'
)
st.plotly_chart(fig, use_container_width=True)

# Por Tipo de Solo
st.write('**Produção por Tipo de Solo**')
fig = px.violin(
    df,
    x='Tipo de Solo',
    y='Produção',
    box=True,
    points='all',
    title='Distribuição de Produção por Tipo de Solo'
)
st.plotly_chart(fig, use_container_width=True)

# ================================================
# 5. Análise de Correlação
# ================================================
st.header('5. Análise de Correlação')

# Mapa de calor de correlação
st.subheader('Mapa de Calor de Correlação')
# Selecionar apenas colunas numéricas
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig_heatmap, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_heatmap)

# ================================================
# 6. Análise Multivariada
# ================================================
st.header('6. Análise Multivariada')

# Pairplot das variáveis numéricas
st.subheader('Pair Plot das Variáveis Numéricas')

# Converter 'Fertilizante' e 'Tipo de Solo' em códigos numéricos para coloração
df_encoded = df.copy()
df_encoded['Fertilizante'] = df_encoded['Fertilizante'].map({'Orgânico': 0, 'Sintético': 1})
df_encoded['Tipo de Solo'] = df_encoded['Tipo de Solo'].map({'Arenoso': 0, 'Argiloso': 1, 'Siltoso': 2})

fig = sns.pairplot(
    df_encoded,
    vars=numeric_columns,
    hue='Fertilizante',
    diag_kind='kde',
    corner=True
)
st.pyplot(fig)

# ================================================
# 7. Análise Interativa
# ================================================
st.header('7. Análise Interativa')

# Seleção de variáveis pelo usuário
st.subheader('Gráfico de Dispersão Interativo')

col1, col2, col3 = st.columns(3)

with col1:
    x_var = st.selectbox('Selecione a variável X:', options=numeric_columns)
with col2:
    y_var = st.selectbox('Selecione a variável Y:', options=numeric_columns)
with col3:
    color_var = st.selectbox('Selecione a variável para colorir:', options=categorical_columns)

corr_temp_prod = df[x_var].corr(df[y_var])
st.write(f"A correlação de Pearson entre {y_var} e {x_var} é: {corr_temp_prod:.2f}")


# Gráfico de dispersão interativo
fig = px.scatter(
    df,
    x=x_var,
    y=y_var,
    color=color_var,
    title=f'{y_var} vs {x_var} colorido por {color_var}'
)
st.plotly_chart(fig, use_container_width=True)