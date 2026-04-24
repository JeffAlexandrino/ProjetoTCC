import pandas as pd
import os

PASTA_DADOS = "data"
PASTA_EMPENHOS = "data/Empenhos"

# LER CSV
df_despesas = pd.read_csv(
    f"{PASTA_DADOS}/despesasOrcamentarias.csv",
    sep=",",
    encoding="latin1"
)

# Limpa colunas
df_despesas.columns = df_despesas.columns.str.strip()

lista_final = []

for _, row in df_despesas.iterrows():

    arquivos_empenho = str(row.get("empenhos", "")).strip()

    if not arquivos_empenho:
        continue

    arquivos_empenho = arquivos_empenho.split(",")

    for arquivo_empenho in arquivos_empenho:
        arquivo_empenho = arquivo_empenho.strip()

        if not arquivo_empenho:
            continue

        caminho_empenho = os.path.join(PASTA_EMPENHOS, arquivo_empenho)

        if not os.path.exists(caminho_empenho):
            print(f"Arquivo não encontrado: {arquivo_empenho}")
            continue

        try:
            df_emp = pd.read_csv(caminho_empenho, sep=",", encoding="latin1")

            for _, emp_row in df_emp.iterrows():
                nova_linha = row.to_dict()

                nova_linha["numeroEmpenho"] = emp_row.get("numeroEmpenho")
                nova_linha["dataEmpenho"] = emp_row.get("dataEmpenho")
                nova_linha["valorEmpenho"] = emp_row.get("valorEmpenho")

                lista_final.append(nova_linha)

        except Exception as e:
            print(f"Erro ao ler {arquivo_empenho}: {e}")

df_final = pd.DataFrame(lista_final)

df_final.to_csv("dataset_final_expandido.csv", index=False)

print("Dataset criado")
print("Linhas:", len(df_final))