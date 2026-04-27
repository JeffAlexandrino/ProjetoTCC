# ProjetoT CC

**Análise e Modelagem de Dados Orçamentários Públicos — Prefeitura de Criciúma/SC**

---

## Visão Geral

Este projeto tem como objetivo analisar dados orçamentários públicos da Prefeitura Municipal de Criciúma/SC, aplicando técnicas de **ciência de dados e aprendizado de máquina** para extrair padrões, identificar comportamentos e gerar insights sobre o uso de recursos públicos.

A proposta vai além da simples integração de dados, incluindo:

- Tratamento e limpeza de dados reais
- Engenharia de atributos (feature engineering)
- Análise exploratória
- Treinamento de modelos de Machine Learning
- Avaliação de desempenho (acurácia, precisão, recall e F1-score)

---

## Objetivos

- Consolidar dados de despesas e empenhos
- Melhorar a qualidade e consistência dos dados
- Criar variáveis relevantes para análise
- Aplicar modelos preditivos
- Avaliar a capacidade dos modelos em identificar padrões nos dados públicos

---

## Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-learn

---

## Estrutura do Projeto

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

## Pipeline do Projeto

O fluxo do projeto segue as etapas abaixo:

### 1. Coleta e Integração
- Leitura dos dados de despesas
- Associação com arquivos de empenhos

### 2. Pré-processamento
- Limpeza de dados inconsistentes
- Tratamento de valores nulos
- Padronização de colunas

### 3. Engenharia de Features
- Criação de variáveis derivadas
- Transformações para melhorar desempenho dos modelos

### 4. Modelagem
- Treinamento de algoritmos de Machine Learning
- Exemplo:
  - Random Forest

### 5. Avaliação
- Métricas utilizadas:
  - Acurácia
  - Precisão
  - Recall
  - F1-score
  - Matriz de confusão

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
````

### 2. Gerar dataset consolidado

```bash
python src/despesasComEmpenhos.py
```

### 3. Executar modelo (se aplicável)

```bash
python src/modelo.py
```

---

## Dados

### Fonte

* Portal da Transparência da Prefeitura de Criciúma/SC

### Características

* Dados reais e públicos
* Encoding original: Latin-1
* Dados tratados para UTF-8

---

## Limitações

* Dados podem conter inconsistências de origem
* Dependência de arquivos externos (empenhos)
* Modelo inicial ainda pode ser otimizado

---

## Autores

* [Jefferson Barzan Alexandrino](https://github.com/JeffAlexandrino)
* [Gabriel Angelo Kaufmann](https://github.com/GabrielAKaufmann)

---

## Referências

* Portal da Transparência de Criciúma
* Lei de Responsabilidade Fiscal
* Dados Abertos do Governo Federal
* Documentação Scikit-learn
