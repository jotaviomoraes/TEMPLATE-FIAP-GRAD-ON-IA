import requests
import pandas as pd
import time
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# =============================================================================
#  CONFIGURAÇÕES GERAIS
# =============================================================================

LATITUDE   = -30.0346
LONGITUDE  = -51.2177
ARQUIVO_CSV = "dataset_queimadas_rs.csv"
JANELA_DIAS = 90        # quantos dias de histórico manter no CSV
HORA_COLETA = "06:00"   # horário diário de atualização automática (hora local)


# =============================================================================
#  UTILITÁRIOS DE DATA
# =============================================================================

DELAY_NASA = 1   # NASA POWER disponibiliza dados com ~1 dia de atraso
DELAY_OM   = 3   # Open-Meteo archive disponibiliza dados com ~3 dias de atraso

def hoje_str_nasa():
    return (datetime.now() - timedelta(days=DELAY_NASA)).strftime("%Y%m%d")

def hoje_str_om():
    return (datetime.now() - timedelta(days=DELAY_OM)).strftime("%Y-%m-%d")

def data_inicio_nasa(dias=JANELA_DIAS):
    return (datetime.now() - timedelta(days=dias)).strftime("%Y%m%d")

def data_inicio_om(dias=JANELA_DIAS):
    return (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")

def ultima_data_no_csv():
    """Retorna a data mais recente já salva no CSV, ou None se o arquivo não existe."""
    if not os.path.exists(ARQUIVO_CSV):
        return None
    try:
        df = pd.read_csv(ARQUIVO_CSV, usecols=["Data"])
        ultima = pd.to_datetime(df["Data"], format="%d/%m/%Y").max()
        return ultima
    except Exception:
        return None


# =============================================================================
#  1. NASA POWER
# =============================================================================

def buscar_dados_nasa(lat, lon, data_inicio, data_fim):
    print(f"  [NASA POWER] Buscando {data_inicio} → {data_fim}...")
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,T2M_MAX,PRECTOTCORR,RH2M,WS2M,WS10M_MAX,ALLSKY_SFC_SW_DWN",
        "community": "ag",
        "longitude": lon,
        "latitude": lat,
        "start": data_inicio,
        "end": data_fim,
        "format": "JSON"
    }
    resposta = requests.get(url, params=params, timeout=60)
    resposta.raise_for_status()
    dados = resposta.json()

    tabela = pd.DataFrame(dados['properties']['parameter']).reset_index()
    tabela = tabela.rename(columns={
        "index"             : "Data",
        "T2M"               : "Temp_Media_C",
        "T2M_MAX"           : "Temp_Max_C",
        "PRECTOTCORR"       : "Chuva_mm",
        "RH2M"              : "Umidade_Ar_pct",
        "WS2M"              : "Vento_2m_ms",
        "WS10M_MAX"         : "Rajada_Max_ms",
        "ALLSKY_SFC_SW_DWN" : "Radiacao_MJ_m2"
    })
    tabela.replace(-999.0, pd.NA, inplace=True)
    print(f"  [NASA POWER] {len(tabela)} registros obtidos.")
    return tabela


# =============================================================================
#  2. OPEN-METEO
# =============================================================================

def buscar_dados_open_meteo(lat, lon, data_inicio, data_fim):
    print(f"  [Open-Meteo] Buscando {data_inicio} → {data_fim}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": data_inicio,
        "end_date"  : data_fim,
        "hourly"    : "relativehumidity_2m,precipitation,windspeed_10m,et0_fao_evapotranspiration"
    }
    resposta = requests.get(url, params=params, timeout=60)
    resposta.raise_for_status()
    dados = resposta.json()

    tabela = pd.DataFrame(dados['hourly'])
    tabela['time'] = pd.to_datetime(tabela['time'])
    tabela['Data'] = tabela['time'].dt.strftime('%Y%m%d')

    agg_rules = {
        "relativehumidity_2m"       : "mean",
        "precipitation"             : "sum",
        "windspeed_10m"             : "mean",
        "et0_fao_evapotranspiration": "sum"
    }
    tabela_diaria = tabela.groupby('Data').agg(agg_rules).reset_index()
    tabela_diaria = tabela_diaria.rename(columns={
        "relativehumidity_2m"       : "UmidAr_OM_pct",
        "precipitation"             : "Chuva_OM_mm",
        "windspeed_10m"             : "Vento_OM_ms",
        "et0_fao_evapotranspiration": "Evapotransp_mm_dia"
    })
    print(f"  [Open-Meteo] {len(tabela_diaria)} registros obtidos.")
    return tabela_diaria


# =============================================================================
#  3. ÍNDICE DE RISCO DE QUEIMADA
# =============================================================================

def calcular_risco_queimada(df):
    df = df.copy()

    def norm(serie, vmin, vmax, inverter=False):
        # Converte pd.NA / NaT para NaN float para que clip/aritmética funcionem
        s = pd.to_numeric(serie, errors='coerce')
        s = (s - vmin) / (vmax - vmin)
        s = s.clip(0, 1)
        # Preenche NaN com 0.5 (valor neutro — não penaliza nem favorece)
        s = s.fillna(0.5)
        return (1 - s) if inverter else s

    fator_temp    = norm(df['Temp_Max_C'],        vmin=5,   vmax=40)
    fator_umid_ar = norm(df['Umidade_Ar_pct'],    vmin=20,  vmax=90,  inverter=True)
    fator_vento   = norm(df['Vento_2m_ms'],       vmin=0,   vmax=10)
    fator_chuva   = norm(df['Chuva_mm'],           vmin=0,   vmax=20,  inverter=True)
    fator_umid_om = norm(df['UmidAr_OM_pct'],      vmin=20,  vmax=90,  inverter=True)
    fator_evap    = norm(df['Evapotransp_mm_dia'], vmin=0,   vmax=6)

    df['Risco_Queimada_0_100'] = (
        fator_temp    * 0.25 +
        fator_umid_ar * 0.20 +
        fator_umid_om * 0.15 +
        fator_vento   * 0.15 +
        fator_chuva   * 0.20 +
        fator_evap    * 0.05
    ) * 100
    # Converte para float nativo antes do round para evitar TypeError com pd.NA
    df['Risco_Queimada_0_100'] = df['Risco_Queimada_0_100'].astype(float).round(1)

    df['Classe_Risco'] = pd.cut(
        df['Risco_Queimada_0_100'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["Baixo", "Moderado", "Alto", "Muito Alto", "Crítico"],
        include_lowest=True
    )
    return df


# =============================================================================
#  4. LÓGICA DE ATUALIZAÇÃO INCREMENTAL
# =============================================================================

COLUNAS_FINAIS = [
    "Data", "Temp_Media_C", "Temp_Max_C", "Chuva_mm",
    "Umidade_Ar_pct", "Vento_2m_ms", "Rajada_Max_ms",
    "Radiacao_MJ_m2", "UmidAr_OM_pct", "Chuva_OM_mm",
    "Vento_OM_ms", "Evapotransp_mm_dia",
    "Risco_Queimada_0_100", "Classe_Risco"
]

def executar_ciclo():
    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  CICLO DE ATUALIZAÇÃO — {agora}")
    print(f"{'='*60}")

    ultima = ultima_data_no_csv()

    # --- Decide o intervalo a buscar ---
    ontem = datetime.now() - timedelta(days=max(DELAY_NASA, DELAY_OM))

    if ultima is None:
        # Primeira execução: busca histórico completo
        print(f"\n  Primeiro carregamento: buscando {JANELA_DIAS} dias de histórico.")
        ini_nasa = data_inicio_nasa()
        ini_om   = data_inicio_om()
    else:
        dias_faltando = (ontem - ultima).days
        if dias_faltando <= 0:
            print(f"\n  Dataset já está atualizado até {ultima.strftime('%d/%m/%Y')}. Nenhuma ação necessária.")
            print(f"  Próxima verificação às {HORA_COLETA}.")
            return
        print(f"\n  Dados desatualizados em {dias_faltando} dia(s). Buscando incremento...")
        # Busca apenas os dias que faltam (+ 1 dia de margem)
        ini_nasa = (ultima + timedelta(days=1)).strftime("%Y%m%d")
        ini_om   = (ultima + timedelta(days=1)).strftime("%Y-%m-%d")

    fim_nasa = hoje_str_nasa()
    fim_om   = hoje_str_om()
    # Garante que fim_om nunca ultrapasse o que a API tem disponível
    print(f"  Período NASA : {ini_nasa} → {fim_nasa}")
    print(f"  Período OM   : {ini_om} → {fim_om}")

    # --- Coleta ---
    try:
        print("\n[1/3] Coletando NASA POWER...")
        df_nasa = buscar_dados_nasa(LATITUDE, LONGITUDE, ini_nasa, fim_nasa)

        print("\n[2/3] Coletando Open-Meteo...")
        df_om = buscar_dados_open_meteo(LATITUDE, LONGITUDE, ini_om, fim_om)
    except requests.RequestException as e:
        print(f"\n  ERRO na requisição: {e}")
        print("  Tentando novamente no próximo ciclo.")
        return

    # --- Fusão e cálculo de risco ---
    print("\n[3/3] Processando novos dados...")
    df_novo = pd.merge(df_nasa, df_om, on="Data", how="inner")
    df_novo = calcular_risco_queimada(df_novo)
    df_novo['Data'] = pd.to_datetime(df_novo['Data'], format='%Y%m%d').dt.strftime('%d/%m/%Y')

    # --- Mescla com histórico existente ---
    if os.path.exists(ARQUIVO_CSV) and ultima is not None:
        df_existente = pd.read_csv(ARQUIVO_CSV, encoding='utf-8-sig')
        df_final = pd.concat([df_existente, df_novo[COLUNAS_FINAIS]], ignore_index=True)

        # Remove duplicatas (mantém o mais recente) e ordena
        df_final['_data_ord'] = pd.to_datetime(df_final['Data'], format='%d/%m/%Y')
        df_final = df_final.drop_duplicates(subset='Data', keep='last')
        df_final = df_final.sort_values('_data_ord').drop(columns='_data_ord')

        # Mantém apenas os últimos JANELA_DIAS dias para não crescer indefinidamente
        df_final = df_final.tail(JANELA_DIAS).reset_index(drop=True)
    else:
        df_final = df_novo[COLUNAS_FINAIS]

    # --- Salva ---
    df_final.to_csv(ARQUIVO_CSV, index=False, encoding='utf-8-sig')

    linhas_novas = len(df_novo)
    print(f"\n  {linhas_novas} novo(s) dia(s) adicionado(s) ao dataset.")
    print(f"  Total no arquivo: {len(df_final)} registros.")
    print(f"  Arquivo salvo em: {ARQUIVO_CSV}")

    # --- Tabela completa ---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_rows', None)

    print(f"\n{'='*60}")
    print("  BASE DE DADOS INTEGRADA: RIO GRANDE DO SUL")
    print(f"{'='*60}")
    print(df_final[COLUNAS_FINAIS].to_string(index=False))

    # --- Resumo estatístico ---
    print(f"\n{'='*60}")
    print("  RESUMO DO PERÍODO")
    print(f"{'='*60}")
    risco_col = pd.to_numeric(df_final['Risco_Queimada_0_100'], errors='coerce')
    chuva_col = pd.to_numeric(df_final['Chuva_mm'], errors='coerce')
    tmax_col  = pd.to_numeric(df_final['Temp_Max_C'], errors='coerce')
    print(f"  Último dia registrado : {df_final.iloc[-1]['Data']}")
    print(f"  Risco médio           : {risco_col.mean():.1f} / 100")
    print(f"  Risco máximo          : {risco_col.max():.1f} / 100")
    print(f"  Dias de alto risco    : {(risco_col >= 60).sum()}")
    print(f"  Chuva acumulada       : {chuva_col.sum():.1f} mm")
    print(f"  Temp. máx. período    : {tmax_col.max():.1f} °C")
    print(f"\n  Distribuição das classes de risco:")
    print(df_final['Classe_Risco'].value_counts().sort_index().to_string())
    print(f"\n  Próxima atualização automática: todos os dias às {HORA_COLETA}.")
    print(f"{'='*60}")


# =============================================================================
#  5. AGENDAMENTO E LOOP PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SISTEMA DE MONITORAMENTO DE QUEIMADAS — RIO GRANDE DO SUL")
    print(f"  Atualização automática agendada: todos os dias às {HORA_COLETA}")
    print("  Pressione Ctrl+C para encerrar.")
    print("=" * 60)

    # Roda imediatamente na inicialização
    executar_ciclo()

    # Aguarda o horário configurado e repete todos os dias
    while True:
        agora = datetime.now()
        hora, minuto = map(int, HORA_COLETA.split(":"))
        proximo = agora.replace(hour=hora, minute=minuto, second=0, microsecond=0)
        if proximo <= agora:
            proximo += timedelta(days=1)
        espera = (proximo - agora).total_seconds()
        proxima_str = proximo.strftime("%d/%m/%Y %H:%M")
        print(f"\n  Aguardando próximo ciclo: {proxima_str}  ({int(espera//3600)}h {int((espera%3600)//60)}min)")
        time.sleep(espera)
        executar_ciclo()