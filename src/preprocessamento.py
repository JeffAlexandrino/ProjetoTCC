import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARREGAR DADOS
df = pd.read_csv("data/despesasComEmpenhos.csv")
df.columns = df.columns.str.strip().str.replace("ï»¿", "", regex=False)

print("Dados carregados:", df.shape)

# 2. LIMPEZA
df = df.dropna(subset=["valorEmpenho", "valorPago"])
df = df.drop_duplicates()

print("Dados limpos:", df.shape)

# 3. TARGET (REGRESSÃO)
df["log_valorPago"] = np.log1p(df["valorPago"])
y_col = "log_valorPago"

# 4. FEATURE ENGINEERING
df["dataEmpenho"] = pd.to_datetime(df["dataEmpenho"], errors="coerce")

df["mes_empenho"]       = df["dataEmpenho"].dt.month
df["trimestre_empenho"] = df["dataEmpenho"].dt.quarter
df["dia_ano_empenho"]   = df["dataEmpenho"].dt.day_of_year
df["dia_semana_empenho"]= df["dataEmpenho"].dt.dayofweek 


df["razao_empenho_orcado"]     = df["valorEmpenhado"]  / (df["valorOrcado"] + 1)
df["razao_liquidado_empenhado"]= df["valorLiquidado"]  / (df["valorEmpenhado"] + 1)
df["razao_orcado_alterado"]    = df["valorOrcadoAlterado"].fillna(0) / (df["valorOrcado"] + 1)
df["variacao_orcamento"]       = df["valorOrcadoAtualizado"] - df["valorOrcado"]

df["log_valorOrcado"]  = np.log1p(df["valorOrcado"])
df["log_valorEmpenho"] = np.log1p(df["valorEmpenho"])
df["log_valorEmpenhado"] = np.log1p(df["valorEmpenhado"])

df["qtd_empenhos"] = df["empenhos"].str.count(r"\d+").fillna(0)

print("\nNovas features criadas")

# 5. FEATURES
features_cat = [
    "descricaoOrgao",
    "descricaoUnidade",
    "descricaoFuncao",
    "descricaoSubfuncao",
    "categoriaEconomica",
    "natureza",
    "descricaoElemento",
    "fontesRecursos",
]

features_num = [
    "mes_empenho",
    "trimestre_empenho",
    "dia_ano_empenho",
    "dia_semana_empenho",
    "razao_empenho_orcado",
    "razao_liquidado_empenhado",
    "razao_orcado_alterado",
    "variacao_orcamento",
    "log_valorOrcado",
    "log_valorEmpenho",
    "log_valorEmpenhado",
    "qtd_empenhos",
    "numeroElemento",
    "idAcao",
]

features_cat = [f for f in features_cat if f in df.columns]
features_num = [f for f in features_num if f in df.columns]
features = features_cat + features_num

print(f"\nTotal de features: {len(features)}")

# 6. SPLIT
treino = df[df["ano"] < 2026].copy()
teste  = df[df["ano"] == 2026].copy()

print(f"\nTreino: {treino.shape} | Teste: {teste.shape}")

# 7. ENCODING
le_dict = {}
for col in features_cat:
    le = LabelEncoder()
    treino[col] = le.fit_transform(treino[col].astype(str))

    test_vals = teste[col].astype(str)
    known = set(le.classes_)
    teste[col] = test_vals.apply(lambda x: le.transform([x])[0] if x in known else -1)

    le_dict[col] = le

print("Encoding feito")

# 8. X e y
X_train = treino[features].fillna(0)
y_train = treino[y_col]

X_test  = teste[features].fillna(0)
y_test  = teste[y_col]

# 9. MODELOS

# Baseline (obrigatório pro TCC!)
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

print("\nModelos treinados")

# 10. AVALIAÇÃO

def avaliar(nome, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n--- {nome} ---")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

# Predições
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

avaliar("Linear Regression (Baseline)", y_test, y_pred_lr)
avaliar("Random Forest", y_test, y_pred_rf)

# 11. IMPORTÂNCIA (RF)
importancias = pd.DataFrame({
    "feature": features,
    "importancia": rf.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\nTop 15 features:")
print(importancias.head(15).to_string(index=False))