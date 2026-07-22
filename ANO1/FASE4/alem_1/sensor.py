import random
from datetime import datetime


def gerar_dado_sensor():
    agora = datetime.now()

    dado = {
        "data_medicao": agora.strftime("%d/%m/%Y"),
        "hora_medicao": agora.strftime("%H:%M:%S"),
        "temperatura": round(random.uniform(18.0, 36.0), 2),
        "umidade": round(random.uniform(35.0, 98.0), 2),
        "pressao": round(random.uniform(948.0, 960.0), 2),
        "precipitacao": round(random.choice([0, 0, 0, 0.2, 0.4, 1.0, 2.5]), 3),
        "vento": round(random.uniform(0.2, 8.0), 2)
    }

    return dado