# 2_Modelagem_Preditiva.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from data_generation import generate_data  # Carrega o dataset

# Configurações gerais
st.set_page_config(page_title='Modelagem Preditiva', layout='wide')

# Título do aplicativo
st.title('Modelagem Preditiva')

# ================================================
# 1. Carregar e Filtrar os Dados
# ================================================

st.header('1. Carregar e Filtrar os Dados')

# Carregar os dados gerados
df = generate_data()

# ----------------------------------
# 1.1. Filtros na Barra Lateral
# ----------------------------------

st.sidebar.title('Filtros de Dados')

# Filtro para Fertilizante
selected_fertilizer = st.sidebar.multiselect(
    'Selecione o Fertilizante:',
    options=df['Fertilizante'].unique(),
    default=df['Fertilizante'].unique()
)

# Filtro para Tipo de Solo
selected_soil = st.sidebar.multiselect(
    'Selecione o Tipo de Solo:',
    options=df['Tipo de Solo'].unique(),
    default=df['Tipo de Solo'].unique()
)

# Filtros para variáveis numéricas

st.sidebar.subheader('Intervalos das Variáveis Numéricas')

# Temperatura
temp_min, temp_max = st.sidebar.slider(
    'Temperatura (°C):',
    min_value=float(df['Temperatura'].min()),
    max_value=float(df['Temperatura'].max()),
    value=(float(df['Temperatura'].min()), float(df['Temperatura'].max()))
)

# Precipitação
precip_min, precip_max = st.sidebar.slider(
    'Precipitação (mm):',
    min_value=float(df['Precipitação'].min()),
    max_value=float(df['Precipitação'].max()),
    value=(float(df['Precipitação'].min()), float(df['Precipitação'].max()))
)

# Umidade
umid_min, umid_max = st.sidebar.slider(
    'Umidade (%):',
    min_value=float(df['Umidade'].min()),
    max_value=float(df['Umidade'].max()),
    value=(float(df['Umidade'].min()), float(df['Umidade'].max()))
)

# ----------------------------------
# 1.2. Aplicar Filtros aos Dados
# ----------------------------------

# Aplicar os filtros selecionados ao DataFrame
df_filtered = df[
    (df['Fertilizante'].isin(selected_fertilizer)) &
    (df['Tipo de Solo'].isin(selected_soil)) &
    (df['Temperatura'] >= temp_min) & (df['Temperatura'] <= temp_max) &
    (df['Precipitação'] >= precip_min) & (df['Precipitação'] <= precip_max) &
    (df['Umidade'] >= umid_min) & (df['Umidade'] <= umid_max)
]

# Verificar se o DataFrame filtrado não está vazio
if df_filtered.empty:
    st.warning('Nenhum dado corresponde aos filtros selecionados. Por favor, ajuste os filtros.')
    st.stop()
else:
    st.subheader('Dados Filtrados')
    st.dataframe(df_filtered.head())

# ================================================
# 2. Preparar os Dados para Modelagem
# ================================================

st.header('2. Preparar os Dados para Modelagem')

# Transformar variáveis categóricas em variáveis dummies
df_ml = pd.get_dummies(df_filtered, columns=['Fertilizante', 'Tipo de Solo'])

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df_ml.drop('Produção', axis=1)
y = df_ml['Produção']

# Verificar se há dados suficientes para treinar o modelo
if len(X) < 2:
    st.warning('Dados insuficientes para treinar o modelo. Por favor, ajuste os filtros para incluir mais dados.')
    st.stop()
else:
    st.write(f'**Total de registros após filtragem:** {len(X)}')

# ================================================
# 3. Treinar o Modelo de Machine Learning
# ================================================

st.header('3. Treinar o Modelo de Machine Learning')

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instanciar o modelo de Random Forest Regressor
model = RandomForestRegressor()

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Avaliar o modelo com os dados de teste
score = model.score(X_test, y_test)
st.write(f'**Acurácia do modelo (R² no conjunto de teste):** {score:.2f}')

# ================================================
# 4. Fazer Previsões com o Modelo
# ================================================

st.header('4. Fazer Previsões com o Modelo')

st.subheader('Insira os Dados para Previsão')

# Coletar entrada do usuário para previsão
temp_input = st.number_input('Temperatura (°C)', value=float(df['Temperatura'].mean()))
precip_input = st.number_input('Precipitação (mm)', value=float(df['Precipitação'].mean()))
umidade_input = st.number_input('Umidade (%)', value=float(df['Umidade'].mean()))
fertilizante_input = st.selectbox('Fertilizante', df['Fertilizante'].unique())
solo_input = st.selectbox('Tipo de Solo', df['Tipo de Solo'].unique())

# ----------------------------------
# 4.1. Validar Entradas do Usuário
# ----------------------------------

# Inicializar variável de controle
input_error = False

# Validar temperatura
if not (-10 <= temp_input <= 50):
    st.error('A temperatura deve estar entre -10°C e 50°C.')
    input_error = True

# Validar precipitação
if not (0 <= precip_input <= 500):
    st.error('A precipitação deve ser entre 0 mm e 500 mm.')
    input_error = True

# Validar umidade
if not (0 <= umidade_input <= 100):
    st.error('A umidade deve ser entre 0% e 100%.')
    input_error = True

# Se não houver erros nas entradas, proceder com a previsão
if not input_error:
    # ----------------------------------
    # 4.2. Preparar os Dados de Entrada
    # ----------------------------------

    # Criar um dicionário com os dados de entrada
    input_data = {
        'Temperatura': [temp_input],
        'Precipitação': [precip_input],
        'Umidade': [umidade_input],
        # Variáveis dummies para Fertilizante
        'Fertilizante_Orgânico': [1 if fertilizante_input == 'Orgânico' else 0],
        'Fertilizante_Sintético': [1 if fertilizante_input == 'Sintético' else 0],
        # Variáveis dummies para Tipo de Solo
        'Tipo de Solo_Arenoso': [1 if solo_input == 'Arenoso' else 0],
        'Tipo de Solo_Argiloso': [1 if solo_input == 'Argiloso' else 0],
        'Tipo de Solo_Siltoso': [1 if solo_input == 'Siltoso' else 0],
    }

    # Converter o dicionário em um DataFrame
    input_df = pd.DataFrame(input_data)

    # Garantir que todas as colunas necessárias estejam presentes
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Adicionar coluna com valor zero

    # Reordenar as colunas para corresponder ao conjunto de treinamento
    input_df = input_df[X.columns]

    # ----------------------------------
    # 4.3. Realizar a Previsão
    # ----------------------------------

    # Fazer a previsão com o modelo treinado
    prediction = model.predict(input_df)

    # Exibir o resultado da previsão
    st.subheader('Resultado da Previsão')
    st.write(f'**Previsão de Produção:** {prediction[0]:.2f} ton/ha')