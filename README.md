# Projeto TCC

## Previsão de Despesas Públicas Municipais Utilizando Técnicas de Machine Learning

### Visão Geral

Este projeto tem como objetivo analisar dados orçamentários públicos da Prefeitura Municipal de Criciúma/SC e aplicar técnicas de Ciência de Dados e Machine Learning para prever valores de despesas públicas a partir de informações financeiras e administrativas presentes nos registros de execução orçamentária.

A pesquisa utiliza dados disponibilizados pelo Portal da Transparência municipal, contemplando etapas de integração, tratamento, análise exploratória, engenharia de atributos e treinamento de modelos preditivos de regressão.

---

## Objetivos

### Objetivo Geral

Desenvolver e avaliar modelos de aprendizado de máquina capazes de prever valores pagos em despesas públicas municipais.

### Objetivos Específicos

* Integrar dados de despesas orçamentárias e empenhos;
* Realizar tratamento e limpeza dos dados;
* Aplicar técnicas de engenharia de atributos;
* Identificar padrões e relações entre variáveis financeiras;
* Treinar modelos de regressão supervisionada;
* Comparar o desempenho dos algoritmos utilizados;
* Avaliar a capacidade preditiva dos modelos por meio de métricas estatísticas.

---

## Tecnologias Utilizadas

* Python 3
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn

---

## Estrutura do Projeto

```text
ProjetoTCC/
│
├── data/
│   ├── despesasComEmpenhos.csv
│   ├── despesasOrcamentarias.csv
│   └── Empenhos/
│       └── empenhos_*.csv
│
├── src/
│   ├── despesasComEmpenhos.py
│   ├── juntarDespesasOrcamentarias.py
│   ├── preprocessamento.py
│   └── modelo.py
│
├── requirements.txt
└── README.md
```

---

## Metodologia

### 1. Coleta e Integração dos Dados

Os dados foram obtidos a partir do Portal da Transparência da Prefeitura de Criciúma/SC e integrados por meio da associação entre registros de despesas e respectivos empenhos.

### 2. Pré-processamento

Foram realizadas atividades de:

* Remoção de inconsistências;
* Tratamento de valores ausentes;
* Conversão de tipos de dados;
* Padronização de formatos;
* Tratamento de outliers utilizando o método IQR com limitação de valores extremos (Winsorização).

### 3. Engenharia de Atributos

Foram criadas variáveis derivadas capazes de fornecer informações adicionais aos modelos, incluindo:

* Diferença entre valor empenhado e valor pago;
* Componentes temporais (ano, mês e trimestre);
* Indicadores financeiros derivados.

### 4. Análise Exploratória dos Dados

Foi realizada análise estatística e visual dos dados para compreensão de distribuições, correlações e padrões relevantes para o processo de modelagem.

### 5. Modelagem Preditiva

Os seguintes algoritmos foram avaliados:

* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

A otimização de hiperparâmetros foi realizada por meio de validação cruzada e Randomized Search.

### 6. Avaliação dos Modelos

Os modelos foram avaliados utilizando métricas adequadas para problemas de regressão:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² (Coeficiente de Determinação)

---

## Principais Resultados

O estudo demonstrou que técnicas de aprendizado de máquina são capazes de identificar padrões relevantes em dados orçamentários públicos, alcançando elevado desempenho preditivo na estimativa dos valores pagos.

Entre os modelos avaliados, o Random Forest apresentou os melhores resultados gerais, obtendo o maior coeficiente de determinação (R²) e os menores erros de previsão.

---

## Como Executar

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Gerar Base Consolidada

```bash
python src/despesasComEmpenhos.py
```

### Realizar Pré-processamento

```bash
python src/preprocessamento.py
```

### Treinar e Avaliar os Modelos

```bash
python src/modelo.py
```

---

## Fonte dos Dados

Portal da Transparência da Prefeitura Municipal de Criciúma/SC.

Características dos dados:

* Dados públicos e oficiais;
* Registros de despesas orçamentárias;
* Informações de empenhos e pagamentos;
* Dados convertidos e tratados para análise computacional.

---

## Limitações

* Dependência da qualidade dos dados disponibilizados pelo portal público;
* Possibilidade de inconsistências originadas na coleta dos dados;
* Resultados limitados ao conjunto de dados analisado;
* Necessidade de revalidação para aplicação em outros municípios.

---

## Autores

* [Jefferson Barzan Alexandrino](https://github.com/JeffAlexandrino)
* [Gabriel Angelo Kaufmann](https://github.com/GabrielAKaufmann)

---

## Referências

* BREIMAN, Leo. Random Forests. Machine Learning, 2001.
* PEDREGOSA, Fabian et al. Scikit-Learn: Machine Learning in Python, 2011.
* CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: A Scalable Tree Boosting System, 2016.
* KE, Guolin et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree, 2017.
* Portal da Transparência da Prefeitura de Criciúma/SC.
