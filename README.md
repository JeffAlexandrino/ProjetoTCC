# ProjetoTCC

**Análise de Dados Orçamentários da Prefeitura Municipal de Criciúma/SC**

---

## Descrição

Este projeto faz parte do Trabalho de Conclusão de Curso (TCC) e tem como objetivo processar, analisar e extrair insights de dados orçamentários públicos da Prefeitura Municipal de Criciúma, Santa Catarina.

O sistema processa dados de despesas orçamentárias e os relaciona com registros de empenhos, permitindo uma visão integrada das movimentações financeiras públicas.

---

## Estrutura do Projeto

```
ProjetoTCC/
├── data/                           # Diretório de dados
│   ├── despesasComEmpenhos.csv     # Despesas combinadas com empenhos
│   ├── despesasOrcamentarias.csv   # Dados originais de despesas
│   └── Empenhos/                   # Arquivos de empenhos individuais
│       └── empenhos_*.csv
├── src/                            # Código fonte
│   ├── despesasComEmpenhos.py      # Script principal: junção despesas + empenhos
│   ├── juntarDespesasOrcamentarias.py # Utilitário: consolidação de CSVs
│   └── preprocessamento.py         # Módulo de pré-processamento (template)
├── requirements.txt                # Dependências Python
└── README.md                       # Este arquivo
```

---

## Como Executar

### 1. Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes)

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o processamento principal

```bash
python src/despesasComEmpenhos.py
```

Este script:
- Lê o arquivo `despesasOrcamentarias.csv`
- Para cada registro, busca os arquivos de empenhos associados
- Gera um novo CSV combinando despesas com dados de empenhos

---

## Dados

### Fonte
- Dados públicos do portal da transparência da Prefeitura de Criciúma/SC

### Formato
- Encoding: Latin-1 (ISO-8859-1)
- Separador: vírgula (,)
- Codificação: UTF-8 para saída

### Estrutura dos Dados

| Campo | Descrição |
|-------|------------|
| `despesasOrcamentarias.csv` | Dados originais de despesas |
| `empenhos_*.csv` | Registros de empenhos por documento |

---

## Dependências

```
pandas>=1.5.0
```

---

## Autores

**Jefferson Barzan Alexandrino**  
**Gabriel Angelo Kaufmann**  

---

## Licença

Este projeto está disponível para fins educacionais.

---

## Referências

- Portal da Transparência - Prefeitura de Criciúma
- Lei de Responsabilidade Fiscal
- Portal de Dados Abertos do Governo Federal