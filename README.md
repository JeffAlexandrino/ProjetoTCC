# Análise de Dados com Machine Learning para Apoio à Tomada de Decisão na Gestão Pública

Este repositório contém o código-fonte e a documentação do projeto de conclusão de curso (TCC) desenvolvido para o Centro Universitário UniSATC. O objetivo do trabalho é prever os valores pagos em despesas orçamentárias da Prefeitura Municipal de Criciúma/SC, servindo como uma ferramenta estratégica de inteligência governamental para transicionar a gestão pública de uma postura reativa para uma baseada em evidências.

---

## Visão Geral do Projeto

A pesquisa consiste em um estudo quantitativo e aplicado utilizando dados financeiros históricos extraídos da plataforma Betha Cloud (Portal da Transparência de Criciúma). Através de um pipeline completo de Ciência de Dados, os registros foram tratados e submetidos a algoritmos de regressão para estimar o desembolso final das despesas a partir de dados orçamentários iniciais.

### Objetivos Específicos
* Consolidar e integrar dados de despesas orçamentárias e seus respectivos empenhos.
* Tratar inconsistências e suavizar outliers orçamentários por meio de Winsorização utilizando o método IQR.
* Codificar variáveis categóricas complexas da administração pública (órgão, função, elemento, fontes de recurso) via *Label Encoding*.
* Implementar validação temporal (*out-of-time*) rigorosa para evitar vazamento de dados (*data leakage*).
* Treinar, otimizar e avaliar modelos preditivos de Machine Learning para predição da variável alvo transformidada em escala logarítmica (`log_valorPago`).

---

## Tecnologias e Bibliotecas Utilizadas

* **Python 3.10+**
* **Pandas & NumPy:** Manipulação de dados, consolidação de arquivos e transformações matemáticas.
* **Scikit-Learn:** Pré-processamento (*LabelEncoder*), algoritmos de Machine Learning, otimização de hiperparâmetros (*RandomizedSearchCV*) e métricas de avaliação.
* **Matplotlib:** Geração de gráficos para análise de importância de atributos.

---

## Estrutura do Repositório

```text
ProjetoTCC/
│
├── data/
│   └── despesasComEmpenhos.csv          # Base integrada e consolidada
│
├── src/
│   ├── despesasComEmpenhos.py           # Script de extração e junção inicial
│   ├── preprocessamento.py              # Limpeza e tratamento de dados
│   └── modelo.py                        # Pipeline de treino, tunagem e avaliação
│
├── requirements.txt                     # Dependências do projeto
└── README.md                            # Documentação principal

```

---

## Metodologia Implementada

1. **Tratamento de Outliers:** Aplicação de Winsorização na variável `valorPago` delimitando os valores num intervalo computado por limites IQR (R$ 0,00 a R$ 64.160.786,12), suavizando valores extremos sem descartar registros legítimos.
2. **Engenharia de Atributos (Feature Engineering):**
* Extração de componentes temporais: `mes_empenho` e `dia_semana`.
* Criação de razões e variações: `razao_empenho_orcado` e `variacao_orcamento`.
* Transformação logarítmica (`np.log1p`) das variáveis monetárias volumosas para estabilização da variância.


3. **Divisão Temporal de Dados:**
* **Treino:** Registros até 2023.
* **Validação interna:** Dados de 2024 (utilizados exclusivamente para seleção de hiperparâmetros por validação cruzada no *RandomizedSearchCV*, funcionando como um *gap* temporal protetivo).
* **Teste (Avaliação Final):** Registros de 2025.
  
---

## Principais Resultados

Os modelos preditivos foram comparados com um *Baseline Ingênuo* (heurística que assume que o valor pago será estritamente igual ao valor orçado). As métricas monetárias foram calculadas revertendo a escala logarítmica:

| Métrica | Baseline Ingênuo | Regressão Linear (Baseline) | Random Forest Regressor |
| --- | --- | --- | --- |
| **MAE (R$)** | R$ 13.540.627,00 | R$ 5.511.745,00 | **R$ 3.403.157,00** |
| **RMSE (R$)** | R$ 36.790.906,00 | R$ 32.281.851,00 | **R$ 5.629.846,00** |
| **$R^2$** | -1,0791 | 0,9129 | **0,9172** |

O **Random Forest Regressor** apresentou uma performance substancialmente superior, reduzindo o erro quadrático em quase **6 vezes** se comparado à Regressão Linear, demonstrando alta robustez contra erros de grande magnitude.

---

## Como Executar o Projeto

1. Instale as dependências necessárias:
```bash
pip install -r requirements.txt

```

2. Processe e prepare a base de dados:
```bash
python src/preprocessamento.py

```

3. Execute o script de modelagem para treinar, otimizar hiperparâmetros e visualizar as avaliações:
```bash
python src/modelo.py

```

---

## Autores

* [Gabriel Angelo Kaufmann](https://www.google.com/search?q=https://github.com/GabrielAKaufmann)
* [Jefferson Barzan Alexandrino](https://www.google.com/search?q=https://github.com/JeffAlexandrino)
