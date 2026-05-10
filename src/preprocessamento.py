import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# 1. CARREGAR DADOS
df = pd.read_csv("data/despesasComEmpenhos.csv", encoding='latin1')
df.columns = df.columns.str.strip().str.replace("ï»¿", "", regex=False)

print(f"Dados carregados: {df.shape}")

# 2. LIMPEZA E TRATAMENTO DE OUTLIERS
df = df.dropna(subset=["valorPago", "valorEmpenho"]).drop_duplicates()

Q1 = df['valorPago'].quantile(0.25)
Q3 = df['valorPago'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

# Aplicando o filtro de outliers 
df = df[df['valorPago'] <= limite_superior].copy()
print(f"Dados após remoção de outliers: {df.shape}")

# 3. TARGET
df["log_valorPago"] = np.log1p(df["valorPago"])
y_col = "log_valorPago"

# 4. FEATURE ENGINEERING
df["dataEmpenho"] = pd.to_datetime(df["dataEmpenho"], errors="coerce")
df["mes_empenho"] = df["dataEmpenho"].dt.month
df["dia_semana"]  = df["dataEmpenho"].dt.dayofweek

# Criando razões apenas com dados de orçamento/empenho
df["razao_empenho_orcado"] = df["valorEmpenho"] / (df["valorOrcado"] + 1)
df["variacao_orcamento"]    = df["valorOrcadoAtualizado"] - df["valorOrcado"]

df["log_valorOrcado"]  = np.log1p(df["valorOrcado"])
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
]

features_num = [
    "mes_empenho",
    "dia_semana",
    "razao_empenho_orcado",
    "variacao_orcamento",
    "log_valorOrcado",
    "log_valorEmpenho",
]

features = features_cat + features_num

# 6. SPLIT - Treino e Teste
treino = df[df["ano"] <= 2023].copy()
teste  = df[df["ano"] >= 2025].copy()

print(f"Treino: {treino.shape} | Teste: {teste.shape}")

# 7. ENCODING
le_dict = {}
for col in features_cat:
    le = LabelEncoder()
    treino[col] = le.fit_transform(treino[col].astype(str))
    
    # Tratando categorias novas no teste
    conhecidos = set(le.classes_)
    teste[col] = teste[col].astype(str).apply(lambda x: le.transform([x])[0] if x in conhecidos else -1)
    le_dict[col] = le

# 8. PREPARAÇÃO X e Y
X_train, y_train = treino[features].fillna(0), treino[y_col]
X_test, y_test   = teste[features].fillna(0), teste[y_col]

# 9. MODELAGEM E OTIMIZAÇÃO
print("\n--- Iniciando Treinamento ---")

# Baseline: Regressão Linear
lr = LinearRegression()
lr.fit(X_train, y_train)

# Otimização: Random Forest com RandomizedSearchCV
param_dist = {
    'n_estimators': [300, 500, 800],
    'max_depth': [20, 40, None],          
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=30,      
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_


# 10. AVALIAÇÃO
def avaliar(nome, y_true, y_pred_log):
    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred_log)
    
    mae  = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2   = r2_score(y_true, y_pred_log) 

    print(f"\n[{nome}]")
    print(f"MAE (Erro Médio em Reais): R$ {mae:,.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

# Baseline ingênuo: predizer que valorPago ≈ valorOrcado
y_pred_naive = X_test["log_valorOrcado"].values
avaliar("Baseline Ingênuo (log_valorOrcado direto)", y_test, y_pred_naive)

avaliar("Linear Regression (Baseline)", y_test, lr.predict(X_test))
avaliar("Random Forest (Otimizado)", y_test, best_rf.predict(X_test))

# Avaliação por faixa de valor
print("\n--- Desempenho por faixa de valor pago ---")
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(best_rf.predict(X_test))

faixas = [
    ("Pequeno (< R$100k)",    y_test_real < 100_000),
    ("Médio (R$100k–1M)",    (y_test_real >= 100_000) & (y_test_real < 1_000_000)),
    ("Grande (> R$1M)",       y_test_real >= 1_000_000),
]

for nome, mask in faixas:
    n = mask.sum()
    if n == 0:
        continue
    mae_f  = mean_absolute_error(y_test_real[mask], y_pred_real[mask])
    r2_f   = r2_score(y_test[mask], best_rf.predict(X_test)[mask])
    print(f"{nome}: n={n}, MAE=R${mae_f:,.0f}, R²={r2_f:.4f}")

# 11. IMPORTÂNCIA DAS FEATURES
importancias = best_rf.feature_importances_
indices = np.argsort(importancias)

plt.figure(figsize=(10, 6))
plt.title('Importância das Variáveis para Previsão de Gastos')
plt.barh(range(len(indices)), importancias[indices], color='skyblue')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importância Relativa')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
