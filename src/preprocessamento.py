import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split
)

df = pd.read_csv(
    "data/despesasComEmpenhos.csv",
    encoding='latin1'
)

df.columns = (
    df.columns
    .str.strip()
    .str.replace("ï»¿", "", regex=False)
)

print(f"Dados carregados: {df.shape}")

df = (
    df
    .dropna(subset=["valorPago", "valorEmpenho"])
    .drop_duplicates()
)

print(f"Dados após limpeza: {df.shape}")

Q1 = df['valorPago'].quantile(0.25)
Q3 = df['valorPago'].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = max(0, Q1 - 1.5 * IQR)
limite_superior = Q3 + 1.5 * IQR

# Substitui valores extremos pelos limites
df['valorPago'] = df['valorPago'].clip(
    lower=limite_inferior,
    upper=limite_superior
)

print("Outliers tratados com winsorização.")

# Target
df["valorPago_boxcox"], lambda_pago = boxcox(
    df["valorPago"] + 1
)

y_col = "valorPago_boxcox"

# Preditoras numéricas
df["valorOrcado_boxcox"], lambda_orcado = boxcox(
    df["valorOrcado"] + 1
)

df["valorEmpenho_boxcox"], lambda_empenho = boxcox(
    df["valorEmpenho"] + 1
)

print("Transformação Box-Cox aplicada.")

df["dataEmpenho"] = pd.to_datetime(
    df["dataEmpenho"],
    errors="coerce"
)

# Variáveis temporais como categóricas
df["mes_empenho"] = (
    df["dataEmpenho"]
    .dt.month
    .astype(str)
)

df["dia_semana"] = (
    df["dataEmpenho"]
    .dt.dayofweek
    .astype(str)
)

# Features derivadas
df["razao_empenho_orcado"] = (
    df["valorEmpenhado"] /
    (df["valorOrcado"] + 1)
)

df["variacao_orcamento"] = (
    df["valorOrcadoAtualizado"] -
    df["valorOrcado"]
)

print("Features criadas.")

features_cat = [
    "descricaoOrgao",
    "descricaoFuncao",
    "categoriaEconomica",
    "natureza",
    "descricaoElemento",
    "fontesRecursos",
    "mes_empenho",
    "dia_semana"
]

features_num = [
    "razao_empenho_orcado",
    "variacao_orcamento",
    "valorOrcado_boxcox",
    "valorEmpenho_boxcox"
]

features = features_cat + features_num

df_modelo = pd.get_dummies(
    df[features],
    columns=features_cat,
    drop_first=True
)

print("Variáveis categóricas codificadas com get_dummies.")

X = df_modelo.fillna(0)
y = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f"Treino: {X_train.shape} | "
    f"Teste: {X_test.shape}"
)

print("\n--- Iniciando Treinamento ---")

lr = LinearRegression()

lr.fit(X_train, y_train)

param_dist = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [2, 5]
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

# =========================================================
# 11. AVALIAÇÃO DOS MODELOS
# =========================================================

def avaliar(nome, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    print(f"\n[{nome}]")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Avaliação Regressão Linear
avaliar(
    "Linear Regression (Baseline)",
    y_test,
    lr.predict(X_test)
)

# Avaliação Random Forest
avaliar(
    "Random Forest (Otimizado)",
    y_test,
    best_rf.predict(X_test)
)

importancias = best_rf.feature_importances_

indices = np.argsort(importancias)

plt.figure(figsize=(12, 8))

plt.title(
    'Importância das Variáveis '
    'para Previsão de Gastos'
)

plt.barh(
    range(len(indices)),
    importancias[indices]
)

plt.yticks(
    range(len(indices)),
    X.columns[indices]
)

plt.xlabel('Importância Relativa')

plt.grid(
    axis='x',
    linestyle='--',
    alpha=0.7
)

plt.tight_layout()

plt.show()