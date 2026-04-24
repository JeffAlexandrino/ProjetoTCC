import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. CARREGAR DADOS
df = pd.read_csv("data/despesasComEmpenhos.csv")
df.columns = df.columns.str.strip().str.replace("ï»¿", "", regex=False)

print("Dados carregados:", df.shape)

# 2. LIMPEZA
df = df.dropna(subset=["valorEmpenho"])
df = df.drop_duplicates()

print("Dados limpos:", df.shape)

# 3. CRIAR TARGET (EFICIÊNCIA DO GASTO)
df["percentual_execucao"] = df["valorPago"] / (df["valorOrcado"] + 1)
df["target"] = np.where(df["percentual_execucao"] >= 0.9, "eficiente", "ineficiente")

print("\nDistribuição do target:")
print(df["target"].value_counts(normalize=True).round(3))

# 4. FEATURE ENGINEERING
# --- Datas ---
df["dataEmpenho"] = pd.to_datetime(df["dataEmpenho"], errors="coerce")
df["mes_empenho"]       = df["dataEmpenho"].dt.month
df["trimestre_empenho"] = df["dataEmpenho"].dt.quarter
df["dia_ano_empenho"]   = df["dataEmpenho"].dt.day_of_year
df["dia_semana_empenho"]= df["dataEmpenho"].dt.dayofweek 

# --- Razões financeiras (sem usar valorPago para evitar leakage!) ---
df["razao_empenho_orcado"]     = df["valorEmpenhado"]  / (df["valorOrcado"] + 1)
df["razao_liquidado_empenhado"]= df["valorLiquidado"]  / (df["valorEmpenhado"] + 1)
df["razao_orcado_alterado"]    = df["valorOrcadoAlterado"].fillna(0) / (df["valorOrcado"] + 1)
df["variacao_orcamento"]       = df["valorOrcadoAtualizado"] - df["valorOrcado"]

# --- Logs de valores absolutos ---
df["log_valorOrcado"]  = np.log1p(df["valorOrcado"])
df["log_valorEmpenho"] = np.log1p(df["valorEmpenho"])
df["log_valorEmpenhado"] = np.log1p(df["valorEmpenhado"])

# --- Contagem de empenhos (proxy de complexidade da despesa) ---
df["qtd_empenhos"] = df["empenhos"].str.count(r"\d+").fillna(0)

print("\nNovas features criadas")

# 5. SELEÇÃO DE FEATURES
features_cat = [
    "descricaoOrgao",      # Órgão responsável
    "descricaoUnidade",    # Unidade gestora
    "descricaoFuncao",     # Função (saúde, educação...)
    "descricaoSubfuncao",  # Subfunção
    "categoriaEconomica",  # Corrente / Capital
    "natureza",            # Natureza da despesa
    "descricaoElemento",   # Elemento de despesa
    "fontesRecursos",      # Fonte de financiamento
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
print("  Categóricas:", features_cat)
print("  Numéricas:  ", features_num)

# 6. SPLIT TREINO/TESTE ANTES de qualquer encoding
treino = df[df["ano"] < 2026].copy()
teste  = df[df["ano"] == 2026].copy()

print(f"\nTreino: {treino.shape} | Teste: {teste.shape}")

# 7. ENCODING — FIT APENAS NO TREINO, TRANSFORM NO TESTE
le_dict = {}
for col in features_cat:
    le = LabelEncoder()
    treino[col] = le.fit_transform(treino[col].astype(str))

    # Categorias novas no teste viram -1
    test_vals = teste[col].astype(str)
    known = set(le.classes_)
    teste[col] = test_vals.apply(lambda x: le.transform([x])[0] if x in known else -1)

    le_dict[col] = le

print("Encoding feito (fit apenas no treino)")

# 8. PREPARAR X e y
X_train = treino[features].fillna(0)
y_train = treino["target"]

X_test  = teste[features].fillna(0)
y_test  = teste["target"]

# 9. MODELO
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("\nModelo treinado")

# 10. PREDIÇÃO E MÉTRICAS
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("RESULTADOS")
print("="*50)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. IMPORTÂNCIA DAS VARIÁVEIS
importancias = pd.DataFrame({
    "feature":    features,
    "importancia": model.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\nTop 15 features mais importantes:")
print(importancias.head(15).to_string(index=False))
