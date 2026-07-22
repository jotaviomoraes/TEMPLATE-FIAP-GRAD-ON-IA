# evolucao.py

import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def avaliar_modelo(modelo, x_teste, y_teste):
    """
    Avalia o desempenho do modelo usando MAE, MSE, RMSE e R².
    """

    previsoes = modelo.predict(x_teste)

    mae = mean_absolute_error(y_teste, previsoes)
    mse = mean_squared_error(y_teste, previsoes)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_teste, previsoes)

    print("\nAvaliação do modelo:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    resultados = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return resultados, previsoes