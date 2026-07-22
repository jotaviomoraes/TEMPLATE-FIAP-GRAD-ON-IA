"""
FarmTech Solutions — Pipeline de Machine Learning Agrícola
Regressão supervisionada com Scikit-Learn.

Modelos treinados:
  1. Rendimento (kg)          — variável-alvo principal
  2. Volume de Irrigação (mm) — quanto de água a cultura precisa
  3. Índice de Fertilização   — necessidade combinada de N, P e K (0–3)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1. CARREGAMENTO E LIMPEZA
# ─────────────────────────────────────────────

def load_data(path: str = "dataset_agricola.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    return df


def treat_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Substitui outliers pela mediana via regra do IQR."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    mediana = df[col].median()
    df[col] = df[col].apply(
        lambda x: mediana if x > q3 + 1.5 * iqr or x < q1 - 1.5 * iqr else x
    )
    return df


# ─────────────────────────────────────────────
#  2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────

NUMERIC_FEATURES   = ["N", "P", "K", "temperatura", "umidade", "ph", "chuva"]
CATEGORICAL_FEATURES = ["cultura", "irrigacao"]
TARGET = "rendimento"

# Features usadas para prever irrigação (sem chuva para não vazar o target)
IRRIG_FEATURES_NUM  = ["N", "P", "K", "temperatura", "umidade", "ph", "rendimento"]
IRRIG_FEATURES_CAT  = ["cultura", "irrigacao"]

# Features para fertilização (sem N, P, K — eles são o alvo)
FERT_FEATURES_NUM   = ["temperatura", "umidade", "ph", "chuva", "rendimento"]
FERT_FEATURES_CAT   = ["cultura", "irrigacao"]


def build_preprocessor(num_feats: list, cat_feats: list) -> ColumnTransformer:
    """StandardScaler + OneHotEncoder configurável por conjunto de features."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])


# ─────────────────────────────────────────────
#  3. HIPERPARÂMETROS — RandomizedSearchCV
# ─────────────────────────────────────────────

# Tamanho da floresta — linspace igual ao notebook
_n_estimators = [int(x) for x in np.linspace(100, 200, num=200)]

# Demais hiperparâmetros investigados
_PARAMS_RF = {
    "model__n_estimators":     _n_estimators,
    "model__max_features":     [None, "sqrt"],
    "model__max_depth":        [int(x) for x in np.linspace(5, 30, num=15)],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf":  [1, 2, 4, 5, 10],
}

_PARAMS_DT = {
    "model__max_depth":         [int(x) for x in np.linspace(3, 20, num=10)],
    "model__max_features":      [None, "sqrt"],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf":  [1, 2, 4, 5, 10],
}


def _tune(pipe: Pipeline, params: dict, X_train, y_train) -> Pipeline:
    """
    RandomizedSearchCV — pega X aleatório de todas as possibilidades
    e traz a melhor combinação (muito mais rápido que GridSearch).
      cv=3      → divide em 3 folds para cross-validation
      n_jobs=-1 → paralelização: usa todos os núcleos do PC
      n_iter=50 → 50 combinações aleatórias (notebook usa 500;
                  reduzido aqui para viabilizar o cache do Streamlit)
    """
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42,
        n_iter=50,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def get_models() -> dict:
    """Modelos base — DT e RF serão otimizados via RandomizedSearchCV em _train_block."""
    return {
        "Regressão Linear":  LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Árvore de Decisão": DecisionTreeRegressor(random_state=42),   # ← tuned abaixo
        "Random Forest":     RandomForestRegressor(random_state=42),   # ← tuned abaixo
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }


# ─────────────────────────────────────────────
#  4. FUNÇÃO GENÉRICA DE TREINO E AVALIAÇÃO
#     (agora inclui MSE além de MAE, RMSE, R²)
# ─────────────────────────────────────────────

def _train_block(X, y, num_feats, cat_feats) -> dict:
    """
    Treina todos os modelos sobre X→y.
    Árvore de Decisão e Random Forest passam por RandomizedSearchCV
    antes de serem avaliados — os demais usam hiperparâmetros fixos.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pre = build_preprocessor(num_feats, cat_feats)
    results = {}

    for name, model in get_models().items():
        pipe = Pipeline([("pre", pre), ("model", model)])

        best_params = {}

        # ── RandomizedSearchCV para Árvore de Decisão e Random Forest ──
        if name == "Árvore de Decisão":
            pipe, best_params = _tune(pipe, _PARAMS_DT, X_train, y_train)
        elif name == "Random Forest":
            pipe, best_params = _tune(pipe, _PARAMS_RF, X_train, y_train)
        else:
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            "pipeline":    pipe,
            "y_test":      y_test,
            "y_pred":      y_pred,
            "best_params": best_params,   # ← hiperparâmetros encontrados pelo RandomSearch
            "MAE":  round(mae,  3),
            "MSE":  round(mse,  3),
            "RMSE": round(rmse, 3),
            "R²":   round(r2,   4),
        }
    return results, X_test, y_test



# ─────────────────────────────────────────────
#  5. TREINO DOS TRÊS MODELOS PREDITIVOS
# ─────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    """
    Retorna três blocos de resultados:
      - results_rend   → previsão de rendimento (kg)
      - results_irrig  → previsão de volume de irrigação (mm)
      - results_fert   → previsão de índice de fertilização (0–3)
    """
    # Limpeza de outliers
    for col in ["temperatura", "umidade", "ph", "chuva", "rendimento"]:
        df = treat_outliers_iqr(df, col)

    # ── TARGET 1: Rendimento ──────────────────────────────────────────────
    X_rend = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_rend = df["rendimento"]
    results_rend, X_test_r, y_test_r = _train_block(
        X_rend, y_rend, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )

    # ── TARGET 2: Volume de Irrigação (chuva em mm como proxy) ───────────
    # Contexto: prever quanto de água (mm) aquela condição exige,
    # usando umidade, temperatura, pH, cultura, irrigação e rendimento
    # como preditores (sem usar chuva como feature, pois é o target).
    X_irrig = df[IRRIG_FEATURES_NUM + IRRIG_FEATURES_CAT]
    y_irrig = df["chuva"]
    results_irrig, X_test_i, y_test_i = _train_block(
        X_irrig, y_irrig, IRRIG_FEATURES_NUM, IRRIG_FEATURES_CAT
    )

    # ── TARGET 3: Índice de Fertilização (N + P + K, 0–3) ────────────────
    # Quantifica a demanda combinada de nutrientes da cultura.
    # Valores mais altos = solo mais carente de adubação.
    df["indice_fertilizacao"] = df["N"] + df["P"] + df["K"]
    X_fert = df[FERT_FEATURES_NUM + FERT_FEATURES_CAT]
    y_fert = df["indice_fertilizacao"]
    results_fert, X_test_f, y_test_f = _train_block(
        X_fert, y_fert, FERT_FEATURES_NUM, FERT_FEATURES_CAT
    )

    return results_rend, results_irrig, results_fert


# ─────────────────────────────────────────────
#  6. IMPORTÂNCIA DE FEATURES (Random Forest)
# ─────────────────────────────────────────────

def get_feature_importance(pipeline: Pipeline, num_feats: list, cat_feats: list) -> pd.DataFrame:
    """
    Extrai importância de features do Random Forest treinado.
    Funciona para qualquer pipeline com OneHotEncoder no pré-processador.
    """
    pre   = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    # Nomes das colunas após OHE
    ohe        = pre.named_transformers_["cat"]
    cat_names  = ohe.get_feature_names_out(cat_feats).tolist()
    all_names  = num_feats + cat_names

    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": all_names, "importance": importances})
    return df_imp.sort_values("importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  7. SUGESTÕES DE MANEJO AGRÍCOLA
# ─────────────────────────────────────────────

def suggest_actions(
    umidade: float,
    ph: float,
    rendimento_previsto: float,
    volume_irrigacao_previsto: float,
    indice_fertilizacao_previsto: float,
    cultura: str,
) -> list[str]:
    """
    Gera recomendações práticas combinando as três previsões dos modelos.
    """
    acoes = []

    # ── Irrigação ─────────────────────────────────────────────────────────
    if umidade < 40:
        acoes.append(
            f"💧 IRRIGAÇÃO URGENTE: umidade em {umidade:.0f}% — "
            f"aplicar aprox. {volume_irrigacao_previsto:.0f} mm de água imediatamente."
        )
    elif umidade < 55:
        acoes.append(
            f"💧 Irrigação moderada recomendada: umidade em {umidade:.0f}% — "
            f"volume sugerido pelo modelo: {volume_irrigacao_previsto:.0f} mm."
        )
    else:
        acoes.append(
            f"✅ Umidade adequada ({umidade:.0f}%) — "
            f"volume de referência do modelo: {volume_irrigacao_previsto:.0f} mm."
        )

    # ── Fertilização ──────────────────────────────────────────────────────
    idx = indice_fertilizacao_previsto
    if idx < 1.0:
        acoes.append(
            f"🌱 Fertilização baixa prevista (índice {idx:.2f}/3) — "
            "solo relativamente rico; monitorar micronutrientes."
        )
    elif idx < 2.0:
        acoes.append(
            f"🧴 Fertilização moderada prevista (índice {idx:.2f}/3) — "
            "considerar reposição de N, P ou K conforme análise de solo."
        )
    else:
        acoes.append(
            f"⚗️  Alta demanda de fertilização prevista (índice {idx:.2f}/3) — "
            "aplicar adubação completa (N + P + K) antes do próximo ciclo."
        )

    # ── pH ────────────────────────────────────────────────────────────────
    if ph < 5.5:
        acoes.append("🧪 pH ácido (< 5,5) — aplicar calcário dolomítico para correção.")
    elif ph > 7.0:
        acoes.append("🧪 pH alcalino (> 7,0) — considerar enxofre agrícola para redução.")
    else:
        acoes.append(f"✅ pH {ph:.2f} dentro da faixa ideal (5,5–7,0).")

    # ── Rendimento ────────────────────────────────────────────────────────
    if rendimento_previsto < 90:
        acoes.append(
            f"⚠️  Rendimento previsto BAIXO ({rendimento_previsto:.1f} kg) — "
            "revisar adubação, irrigação e condições climáticas urgentemente."
        )
    elif rendimento_previsto < 140:
        acoes.append(
            f"📊 Rendimento previsto MODERADO ({rendimento_previsto:.1f} kg) — "
            "manter manejo atual com monitoramento contínuo."
        )
    else:
        acoes.append(
            f"🌾 Rendimento previsto EXCELENTE ({rendimento_previsto:.1f} kg) — "
            "continuar as práticas atuais de manejo."
        )

    # ── Dica específica por cultura ───────────────────────────────────────
    dicas = {
        "cafe":  "☕ Café: manter temperatura entre 18–23 °C e sombreamento adequado.",
        "soja":  "🫘 Soja: atenção ao fotoperíodo; chuva > 200 mm exige drenagem eficiente.",
        "milho": "🌽 Milho: garantir K disponível na fase de enchimento de grãos.",
        "trigo": "🌾 Trigo: monitorar fungos em umidade > 80%; ventilação é essencial.",
    }
    acoes.append(dicas.get(cultura, "🌱 Monitorar condições locais regularmente."))

    return acoes


if __name__ == "__main__":
    df = load_data("dataset_agricola.csv")
    res_rend, res_irrig, res_fert = train_and_evaluate(df)

    for titulo, bloco in [
        ("RENDIMENTO (kg)", res_rend),
        ("VOLUME IRRIGAÇÃO (mm)", res_irrig),
        ("ÍNDICE FERTILIZAÇÃO (0–3)", res_fert),
    ]:
        print(f"\n📊 {titulo}\n" + "─" * 55)
        for name, m in bloco.items():
            print(
                f"  {name:25s} | MAE={m['MAE']:.3f} | MSE={m['MSE']:.3f} "
                f"| RMSE={m['RMSE']:.3f} | R²={m['R²']:.4f}"
            )
