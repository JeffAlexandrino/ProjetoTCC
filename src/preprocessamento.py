import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# 1. CARREGAR DADOS
df = pd.read_csv("data/despesasComEmpenhos.csv", encoding='latin1')
df.columns = df.columns.str.strip().str.replace("ï»¿", "", regex=False)

print(f"Dados carregados: {df.shape}")

# 2. LIMPEZA E TRATAMENTO DE OUTLIERS
df = df.dropna(subset=["valorPago", "valorEmpenho"]).drop_duplicates()

# Cálculo do IQR
Q1 = df['valorPago'].quantile(0.25)
Q3 = df['valorPago'].quantile(0.75)
IQR = Q3 - Q1

limite_superior = Q3 + 1.5 * IQR
limite_inferior = max(0, Q1 - 1.5 * IQR)

# Winsorização (cap dos valores extremos)
df['valorPago'] = df['valorPago'].clip(
    lower=limite_inferior,
    upper=limite_superior
)

print("Outliers tratados com winsorização.")

# 3. TARGET (TRANSFORMAÇÃO LOGARÍTMICA)
df["log_valorPago"] = np.log1p(df["valorPago"])
y_col = "log_valorPago"

# 4. FEATURE ENGINEERING
df["dataEmpenho"] = pd.to_datetime(df["dataEmpenho"], errors="coerce")

# Agora como categóricas
df["mes_empenho"] = df["dataEmpenho"].dt.month.astype(str)
df["dia_semana"] = df["dataEmpenho"].dt.dayofweek.astype(str)

# Features derivadas
df["razao_empenho_orcado"] = (
    df["valorEmpenhado"] / (df["valorOrcado"] + 1)
)

df["variacao_orcamento"] = (
    df["valorOrcadoAtualizado"] - df["valorOrcado"]
)

df["log_valorOrcado"] = np.log1p(df["valorOrcado"])
df["log_valorEmpenho"] = np.log1p(df["valorEmpenho"])

print("Features criadas.")

# 5. DEFINIÇÃO DE VARIÁVEIS

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
    "log_valorEmpenho",
]

features = features_cat + features_num

# 6. ENCODING DAS VARIÁVEIS CATEGÓRICAS

le_dict = {}

for col in features_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# 7. PREPARAÇÃO X E Y

X = df[features].fillna(0)
y = df[y_col]

# Split padrão 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

# 8. MODELAGEM

print("\n--- Iniciando Treinamento ---")

# Baseline
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest otimizado
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

# 9. AVALIAÇÃO

def avaliar(nome, y_true, y_pred_log):

    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true_real, y_pred_real)

    rmse = np.sqrt(
        mean_squared_error(y_true_real, y_pred_real)
    )

    r2 = r2_score(y_true, y_pred_log)

    print(f"\n[{nome}]")
    print(f"MAE: R$ {mae:,.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

avaliar(
    "Linear Regression (Baseline)",
    y_test,
    lr.predict(X_test)
)

avaliar(
    "Random Forest (Otimizado)",
    y_test,
    best_rf.predict(X_test)
)

# 10. IMPORTÂNCIA DAS FEATURES

importancias = best_rf.feature_importances_
indices = np.argsort(importancias)

plt.figure(figsize=(10, 6))

plt.title('Importância das Variáveis para Previsão de Gastos')

plt.barh(
    range(len(indices)),
    importancias[indices]
)

plt.yticks(
    range(len(indices)),
    [features[i] for i in indices]
)

plt.xlabel('Importância Relativa')

plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()