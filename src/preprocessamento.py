import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# 1. CARREGAMENTO
df = pd.read_csv("data/despesasComEmpenhos.csv", encoding="latin1")
df.columns = df.columns.str.strip().str.replace("ï»¿", "", regex=False)

print(f"Dados carregados: {df.shape}")

# 2. LIMPEZA
df = df.dropna(subset=["valorPago", "valorEmpenho"])
df = df.drop_duplicates()

# 3. TRATAMENTO DE OUTLIERS (WINSORIZAÇÃO)
Q1 = df["valorPago"].quantile(0.25)
Q3 = df["valorPago"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = max(0, Q1 - 1.5 * IQR)
limite_superior = Q3 + 1.5 * IQR

df["valorPago"] = df["valorPago"].clip(
    lower=limite_inferior,
    upper=limite_superior
)

print(
    f"Outliers suavizados "
    f"({limite_inferior:,.2f} até {limite_superior:,.2f})"
)

# 4. TARGET
df["log_valorPago"] = np.log1p(df["valorPago"])
y_col = "log_valorPago"

# 5. FEATURE ENGINEERING
df["dataEmpenho"] = pd.to_datetime(
    df["dataEmpenho"],
    errors="coerce"
)

df["mes_empenho"] = df["dataEmpenho"].dt.month
df["dia_semana"] = df["dataEmpenho"].dt.dayofweek

df["razao_empenho_orcado"] = (
    df["valorEmpenho"] /
    (df["valorOrcado"] + 1)
)

df["variacao_orcamento"] = (
    df["valorOrcadoAtualizado"]
    - df["valorOrcado"]
)

df["log_valorOrcado"] = np.log1p(df["valorOrcado"])
df["log_valorEmpenho"] = np.log1p(df["valorEmpenho"])

print("Features criadas.")

# 6. DEFINIÇÃO DAS FEATURES
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
    "log_valorOrcado",
    "log_valorEmpenho"
]

features = features_cat + features_num

# 7. SPLIT TEMPORAL
treino = df[df["ano"] <= 2023].copy()
validacao = df[df["ano"] == 2024].copy()
teste = df[df["ano"] == 2025].copy()

print(
    f"Treino: {treino.shape} | "
    f"Validação: {validacao.shape} | "
    f"Teste: {teste.shape}"
)

# 8. ENCODING
le_dict = {}

for col in features_cat:

    le = LabelEncoder()

    treino[col] = le.fit_transform(
        treino[col].astype(str)
    )

    conhecidos = set(le.classes_)

    validacao[col] = validacao[col].astype(str).apply(
        lambda x:
        le.transform([x])[0]
        if x in conhecidos
        else -1
    )

    teste[col] = teste[col].astype(str).apply(
        lambda x:
        le.transform([x])[0]
        if x in conhecidos
        else -1
    )

    le_dict[col] = le

# 9. MATRIZES
X_train = treino[features].fillna(0)
y_train = treino[y_col]

X_val = validacao[features].fillna(0)
y_val = validacao[y_col]

X_test = teste[features].fillna(0)
y_test = teste[y_col]

# 10. MODELOS
print("\n--- Iniciando Treinamento ---")

# Baseline Linear
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest
param_dist = {
    "n_estimators": [300, 500, 800],
    "max_depth": [20, 40, None],
    "min_samples_leaf": [1, 2, 5],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5]
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print("\nMelhores parâmetros:")
print(rf_search.best_params_)

# 11. AVALIAÇÃO
def avaliar(nome, y_true, y_pred_log):

    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred_log)

    mae = mean_absolute_error(
        y_true_real,
        y_pred_real
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true_real,
            y_pred_real
        )
    )

    r2 = r2_score(
        y_true,
        y_pred_log
    )

    print(f"\n[{nome}]")
    print(f"MAE: R$ {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}")

# Baseline ingênuo
y_pred_naive = X_test["log_valorOrcado"]

avaliar(
    "Baseline Ingênuo",
    y_test,
    y_pred_naive
)

# Linear
avaliar(
    "Linear Regression",
    y_test,
    lr.predict(X_test)
)

# Random Forest
avaliar(
    "Random Forest",
    y_test,
    best_rf.predict(X_test)
)

# 12. AVALIAÇÃO POR FAIXA
print("\n--- Desempenho por faixa ---")

y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(best_rf.predict(X_test))

faixas = [
    ("Pequeno (<100k)", y_test_real < 100_000),
    (
        "Médio (100k-1M)",
        (y_test_real >= 100_000)
        & (y_test_real < 1_000_000)
    ),
    ("Grande (>1M)", y_test_real >= 1_000_000)
]

for nome, mask in faixas:

    if mask.sum() == 0:
        continue

    mae = mean_absolute_error(
        y_test_real[mask],
        y_pred_real[mask]
    )

    r2 = r2_score(
        y_test[mask],
        best_rf.predict(X_test)[mask]
    )

    print(
        f"{nome}: "
        f"n={mask.sum()} | "
        f"MAE=R${mae:,.0f} | "
        f"R²={r2:.4f}"
    )

# 13. IMPORTÂNCIA DAS FEATURES
importancias = best_rf.feature_importances_
indices = np.argsort(importancias)

plt.figure(figsize=(10, 6))

plt.title(
    "Importância das Variáveis para Previsão de Gastos"
)

plt.barh(
    range(len(indices)),
    importancias[indices]
)

plt.yticks(
    range(len(indices)),
    [features[i] for i in indices]
)

plt.xlabel("Importância Relativa")

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()
plt.show()
