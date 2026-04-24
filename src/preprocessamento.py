# """
# Módulo de pré-processamento de dados orçamentários
# Autor: TCC - Engenharia de Software
# Descrição: Responsável pela limpeza, transformação e preparação dos dados
#            para os modelos de machine learning
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import logging
# from typing import Tuple, Optional, List, Dict
# from sklearn.preprocessing import MinMaxScaler

# # Configuração de logging para acompanhamento do processo
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class PreprocessadorDadosOrcamentarios:
#     """
#     Classe responsável pelo pré-processamento dos dados orçamentários
#     da Prefeitura Municipal de Criciúma/SC
#     """
    
#     def __init__(self):
#         """Inicializa o preprocessador com as configurações padrão"""
#         self.scaler = MinMaxScaler()
#         self.colunas_removidas = []
#         self.dados_originais_shape = None
#         self.dados_processados_shape = None
        
#         # Definição das colunas consideradas essenciais para análise
#         self.colunas_essenciais = [
#             'data_referencia',
#             'categoria_orcamentaria',
#             'unidade_gestora',
#             'classificacao_orcamentaria'
#         ]
        
#         # Colunas numéricas que serão utilizadas nos modelos
#         self.colunas_numericas = [
#             'valor_arrecadado',  # receitas
#             'valor_executado'    # despesas
#         ]
        
#         logger.info("Preprocessador inicializado com sucesso")
    
#     def carregar_dados(self, caminho_arquivo: str, tipo_arquivo: str = 'auto') -> pd.DataFrame:
#         """
#         Carrega dados de arquivo CSV ou Excel
        
#         Args:
#             caminho_arquivo: Caminho do arquivo a ser carregado
#             tipo_arquivo: 'csv', 'excel' ou 'auto' (detecta automaticamente)
        
#         Returns:
#             DataFrame com os dados carregados
#         """
#         logger.info(f"Carregando arquivo: {caminho_arquivo}")
        
#         try:
#             # Verifica se o arquivo existe
#             if not Path(caminho_arquivo).exists():
#                 raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
            
#             # Detecta ou usa o tipo especificado
#             if tipo_arquivo == 'auto':
#                 extensao = Path(caminho_arquivo).suffix.lower()
#                 if extensao in ['.csv']:
#                     tipo_arquivo = 'csv'
#                 elif extensao in ['.xlsx', '.xls']:
#                     tipo_arquivo = 'excel'
#                 else:
#                     raise ValueError(f"Tipo de arquivo não suportado: {extensao}")
            
#             # Carrega conforme o tipo
#             if tipo_arquivo == 'csv':
#                 df = pd.read_csv(caminho_arquivo)
#             elif tipo_arquivo == 'excel':
#                 df = pd.read_excel(caminho_arquivo)
#             else:
#                 raise ValueError(f"Tipo de arquivo inválido: {tipo_arquivo}")
            
#             logger.info(f"Dados carregados com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas")
#             self.dados_originais_shape = df.shape
#             return df
            
#         except Exception as e:
#             logger.error(f"Erro ao carregar arquivo: {str(e)}")
#             raise
    
#     def validar_colunas(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Valida e padroniza os nomes das colunas para o formato esperado
        
#         Args:
#             df: DataFrame original
        
#         Returns:
#             DataFrame com colunas padronizadas
#         """
#         logger.info("Validando e padronizando nomes das colunas")
        
#         # Dicionário de mapeamento para padronização
#         mapeamento_colunas = {
#             # Colunas de data
#             'data': 'data_referencia',
#             'dt_referencia': 'data_referencia',
#             'periodo': 'data_referencia',
#             'ano_mes': 'data_referencia',
            
#             # Colunas de categoria
#             'categoria': 'categoria_orcamentaria',
#             'categoria_orc': 'categoria_orcamentaria',
#             'rubrica': 'categoria_orcamentaria',
            
#             # Colunas de unidade
#             'unidade': 'unidade_gestora',
#             'orgao': 'unidade_gestora',
#             'secretaria': 'unidade_gestora',
            
#             # Colunas de classificação
#             'classificacao': 'classificacao_orcamentaria',
#             'natureza': 'classificacao_orcamentaria',
            
#             # Colunas de valores
#             'receita': 'valor_arrecadado',
#             'receitas': 'valor_arrecadado',
#             'arrecadado': 'valor_arrecadado',
#             'valor_receita': 'valor_arrecadado',
            
#             'despesa': 'valor_executado',
#             'despesas': 'valor_executado',
#             'executado': 'valor_executado',
#             'valor_despesa': 'valor_executado'
#         }
        
#         # Renomeia colunas existentes
#         colunas_existentes = {}
#         for col in df.columns:
#             col_lower = col.lower().strip()
#             if col_lower in mapeamento_colunas:
#                 colunas_existentes[col] = mapeamento_colunas[col_lower]
        
#         if colunas_existentes:
#             df = df.rename(columns=colunas_existentes)
#             logger.info(f"Colunas renomeadas: {colunas_existentes}")
        
#         return df
    
#     def tratar_dados_ausentes(self, df: pd.DataFrame, strategy: str = 'mediana') -> pd.DataFrame:
#         """
#         Trata valores ausentes no DataFrame
        
#         Args:
#             df: DataFrame a ser tratado
#             strategy: Estratégia de preenchimento ('mediana', 'media', 'zero', 'remover')
        
#         Returns:
#             DataFrame com valores ausentes tratados
#         """
#         logger.info("Tratando valores ausentes")
        
#         # Conta valores ausentes por coluna
#         missing_before = df.isnull().sum()
#         logger.info(f"Valores ausentes antes do tratamento:\n{missing_before[missing_before > 0]}")
        
#         # Trata colunas numéricas
#         for col in self.colunas_numericas:
#             if col in df.columns and df[col].isnull().any():
#                 if strategy == 'mediana':
#                     valor_preenchimento = df[col].median()
#                 elif strategy == 'media':
#                     valor_preenchimento = df[col].mean()
#                 elif strategy == 'zero':
#                     valor_preenchimento = 0
#                 else:
#                     continue
                
#                 df[col].fillna(valor_preenchimento, inplace=True)
#                 logger.info(f"Coluna '{col}': preenchidos {missing_before[col]} valores com {strategy} = {valor_preenchimento:.2f}")
        
#         # Remove linhas onde colunas essenciais estão ausentes
#         for col in self.colunas_essenciais:
#             if col in df.columns:
#                 antes = len(df)
#                 df = df.dropna(subset=[col])
#                 if antes > len(df):
#                     logger.info(f"Removidas {antes - len(df)} linhas com coluna '{col}' ausente")
        
#         # Verifica se ainda há valores ausentes
#         missing_after = df.isnull().sum()
#         if missing_after.sum() > 0:
#             logger.warning(f"Ainda há valores ausentes:\n{missing_after[missing_after > 0]}")
#         else:
#             logger.info("Todos os valores ausentes foram tratados com sucesso")
        
#         return df
    
#     def remover_duplicatas(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
#         """
#         Remove registros duplicados do DataFrame
        
#         Args:
#             df: DataFrame a ser tratado
#             subset: Lista de colunas para verificar duplicatas (None = todas)
        
#         Returns:
#             DataFrame sem duplicatas
#         """
#         logger.info("Removendo registros duplicados")
        
#         antes = len(df)
        
#         if subset is None:
#             # Usa todas as colunas exceto possíveis IDs
#             colunas_para_check = [col for col in df.columns if 'id' not in col.lower()]
#             df = df.drop_duplicates(subset=colunas_para_check)
#         else:
#             df = df.drop_duplicates(subset=subset)
        
#         depois = len(df)
#         removidas = antes - depois
        
#         if removidas > 0:
#             logger.info(f"Removidas {removidas} linhas duplicadas ({removidas/antes*100:.2f}% dos dados)")
#         else:
#             logger.info("Nenhuma linha duplicada encontrada")
        
#         return df
    
#     def padronizar_datas(self, df: pd.DataFrame, coluna_data: str = 'data_referencia') -> pd.DataFrame:
#         """
#         Padroniza o formato das datas e extrai features temporais
        
#         Args:
#             df: DataFrame a ser tratado
#             coluna_data: Nome da coluna de data
        
#         Returns:
#             DataFrame com datas padronizadas e features temporais
#         """
#         logger.info(f"Padronizando datas na coluna: {coluna_data}")
        
#         if coluna_data not in df.columns:
#             logger.warning(f"Coluna '{coluna_data}' não encontrada. Pulando padronização de datas")
#             return df
        
#         # Tenta converter para datetime com diferentes formatos
#         formatos_tentativa = [
#             '%Y-%m-%d',     # 2024-01-15
#             '%d/%m/%Y',     # 15/01/2024
#             '%Y%m%d',       # 20240115
#             '%d-%m-%Y',     # 15-01-2024
#             '%Y-%m',        # 2024-01
#             '%m/%Y',        # 01/2024
#         ]
        
#         for formato in formatos_tentativa:
#             try:
#                 df[coluna_data] = pd.to_datetime(df[coluna_data], format=formato, errors='ignore')
#                 break
#             except:
#                 continue
        
#         # Se ainda não for datetime, tenta conversão automática
#         if not pd.api.types.is_datetime64_any_dtype(df[coluna_data]):
#             df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
        
#         # Extrai features temporais úteis para análise
#         df['ano'] = df[coluna_data].dt.year
#         df['mes'] = df[coluna_data].dt.month
#         df['trimestre'] = df[coluna_data].dt.quarter
#         df['semestre'] = df[coluna_data].dt.month.apply(lambda x: 1 if x <= 6 else 2)
        
#         # Remove linhas com data inválida
#         antes = len(df)
#         df = df.dropna(subset=[coluna_data])
#         depois = len(df)
        
#         if antes > depois:
#             logger.info(f"Removidas {antes - depois} linhas com datas inválidas")
        
#         logger.info(f"Período dos dados: {df[coluna_data].min()} a {df[coluna_data].max()}")
#         logger.info(f"Features temporais extraídas: ano, mes, trimestre, semestre")
        
#         return df
    
#     def normalizar_valores(self, df: pd.DataFrame, colunas: Optional[List[str]] = None) -> pd.DataFrame:
#         """
#         Normaliza valores numéricos usando Min-Max Scaling
        
#         Args:
#             df: DataFrame a ser tratado
#             colunas: Lista de colunas para normalizar (None = todas numéricas)
        
#         Returns:
#             DataFrame com valores normalizados
#         """
#         logger.info("Normalizando valores numéricos (Min-Max Scaling)")
        
#         if colunas is None:
#             colunas = self.colunas_numericas
        
#         # Filtra apenas colunas existentes
#         colunas_para_normalizar = [col for col in colunas if col in df.columns]
        
#         if not colunas_para_normalizar:
#             logger.warning("Nenhuma coluna numérica encontrada para normalização")
#             return df
        
#         logger.info(f"Normalizando colunas: {colunas_para_normalizar}")
        
#         # Salva estatísticas antes da normalização
#         for col in colunas_para_normalizar:
#             logger.info(f"Coluna '{col}' - Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Média: {df[col].mean():.2f}")
        
#         # Aplica MinMaxScaler
#         df_normalizado = df.copy()
#         valores_normalizados = self.scaler.fit_transform(df[colunas_para_normalizar])
        
#         # Substitui os valores originais pelos normalizados
#         for i, col in enumerate(colunas_para_normalizar):
#             df_normalizado[f'{col}_normalizado'] = valores_normalizados[:, i]
#             logger.info(f"Coluna '{col}_normalizado' criada com valores normalizados")
        
#         return df_normalizado
    
#     def selecionar_atributos_relevantes(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Seleciona apenas os atributos relevantes para o modelo preditivo
        
#         Args:
#             df: DataFrame original
        
#         Returns:
#             DataFrame com atributos selecionados
#         """
#         logger.info("Selecionando atributos relevantes para o modelo")
        
#         colunas_relevantes = [
#             'data_referencia',
#             'ano',
#             'mes',
#             'trimestre',
#             'semestre',
#             'categoria_orcamentaria',
#             'unidade_gestora',
#             'classificacao_orcamentaria',
#             'valor_arrecadado_normalizado',
#             'valor_executado_normalizado'
#         ]
        
#         # Mantém colunas que existem no DataFrame
#         colunas_existentes = [col for col in colunas_relevantes if col in df.columns]
        
#         # Adiciona colunas originais se normalizadas não existirem
#         if 'valor_arrecadado_normalizado' not in colunas_existentes and 'valor_arrecadado' in df.columns:
#             colunas_existentes.append('valor_arrecadado')
        
#         if 'valor_executado_normalizado' not in colunas_existentes and 'valor_executado' in df.columns:
#             colunas_existentes.append('valor_executado')
        
#         df_selecionado = df[colunas_existentes].copy()
        
#         # Converte colunas categóricas para tipo category (otimiza memória)
#         colunas_categoricas = ['categoria_orcamentaria', 'unidade_gestora', 'classificacao_orcamentaria']
#         for col in colunas_categoricas:
#             if col in df_selecionado.columns:
#                 df_selecionado[col] = df_selecionado[col].astype('category')
        
#         logger.info(f"Atributos selecionados: {list(df_selecionado.columns)}")
#         logger.info(f"Dados após seleção: {df_selecionado.shape[0]} linhas e {df_selecionado.shape[1]} colunas")
        
#         self.dados_processados_shape = df_selecionado.shape
#         return df_selecionado
    
#     def gerar_relatorio_preprocessamento(self, df_original: pd.DataFrame, df_processado: pd.DataFrame) -> Dict:
#         """
#         Gera um relatório detalhado do pré-processamento realizado
        
#         Args:
#             df_original: DataFrame original
#             df_processado: DataFrame após pré-processamento
        
#         Returns:
#             Dicionário com métricas do pré-processamento
#         """
#         logger.info("Gerando relatório de pré-processamento")
        
#         relatorio = {
#             'dados_originais': {
#                 'linhas': df_original.shape[0],
#                 'colunas': df_original.shape[1],
#                 'colunas_nomes': list(df_original.columns)
#             },
#             'dados_processados': {
#                 'linhas': df_processado.shape[0],
#                 'colunas': df_processado.shape[1],
#                 'colunas_nomes': list(df_processado.columns)
#             },
#             'transformacoes': {
#                 'linhas_removidas': df_original.shape[0] - df_processado.shape[0],
#                 'percentual_reducao': ((df_original.shape[0] - df_processado.shape[0]) / df_original.shape[0]) * 100
#             }
#         }
        
#         # Adiciona informações sobre valores ausentes
#         if df_original.isnull().sum().sum() > 0:
#             relatorio['valores_ausentes_original'] = df_original.isnull().sum().to_dict()
        
#         if df_processado.isnull().sum().sum() > 0:
#             relatorio['valores_ausentes_processado'] = df_processado.isnull().sum().to_dict()
        
#         logger.info(f"Relatório gerado: {relatorio['transformacoes']['linhas_removidas']} linhas removidas "
#                    f"({relatorio['transformacoes']['percentual_reducao']:.2f}%)")
        
#         return relatorio
    
#     def executar_preprocessamento_completo(
#         self, 
#         caminho_arquivo: str, 
#         tipo_arquivo: str = 'auto',
#         salvar_processado: bool = True,
#         caminho_saida: str = 'data/dados_processados.csv'
#     ) -> Tuple[pd.DataFrame, Dict]:
#         """
#         Executa o pipeline completo de pré-processamento
        
#         Args:
#             caminho_arquivo: Caminho do arquivo original
#             tipo_arquivo: Tipo do arquivo ('csv', 'excel', 'auto')
#             salvar_processado: Se deve salvar o DataFrame processado
#             caminho_saida: Caminho para salvar os dados processados
        
#         Returns:
#             Tupla com (DataFrame processado, relatório)
#         """
#         logger.info("="*60)
#         logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
#         logger.info("="*60)
        
#         # Carrega dados
#         df = self.carregar_dados(caminho_arquivo, tipo_arquivo)
        
#         # Valida e padroniza colunas
#         df = self.validar_colunas(df)
        
#         # Remove duplicatas
#         df = self.remover_duplicatas(df)
        
#         # Padroniza datas
#         df = self.padronizar_datas(df)
        
#         # Trata valores ausentes
#         df = self.tratar_dados_ausentes(df)
        
#         # Normaliza valores
#         df = self.normalizar_valores(df)
        
#         # Seleciona atributos relevantes
#         df_final = self.selecionar_atributos_relevantes(df)
        
#         # Gera relatório
#         relatorio = self.gerar_relatorio_preprocessamento(df, df_final)
        
#         # Salva dados processados
#         if salvar_processado:
#             Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
#             df_final.to_csv(caminho_saida, index=False, encoding='utf-8-sig')
#             logger.info(f"Dados processados salvos em: {caminho_saida}")
        
#         logger.info("="*60)
#         logger.info("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO")
#         logger.info("="*60)
        
#         return df_final, relatorio


# # Função de exemplo para demonstração
# def demonstrar_preprocessamento():
#     """
#     Função demonstrativa que cria dados de exemplo e executa o pré-processamento
#     """
    
#     # Criando dados de exemplo simulando o formato da Betha Cloud
#     print("Criando dados de exemplo para demonstração...")
    
#     np.random.seed(42)
#     datas = pd.date_range('2023-01-01', '2025-12-31', freq='M')
    
#     dados_exemplo = []
#     categorias = ['Educação', 'Saúde', 'Infraestrutura', 'Segurança']
#     unidades = ['Secretaria A', 'Secretaria B', 'Secretaria C']
#     classificacoes = ['Corrente', 'Capital']
    
#     for data in datas:
#         for categoria in categorias:
#             for unidade in unidades[:2]:  # Simplificando
#                 for classificacao in classificacoes:
#                     # Gera valores com algumas variações e dados ausentes
#                     receita = np.random.normal(100000, 20000)
#                     despesa = np.random.normal(80000, 15000)
                    
#                     # Introduz alguns valores ausentes (5% de chance)
#                     if np.random.random() < 0.05:
#                         receita = np.nan
#                     if np.random.random() < 0.05:
#                         despesa = np.nan
                    
#                     dados_exemplo.append({
#                         'data_referencia': data,
#                         'categoria_orcamentaria': categoria,
#                         'unidade_gestora': unidade,
#                         'classificacao_orcamentaria': classificacao,
#                         'valor_arrecadado': receita,
#                         'valor_executado': despesa
#                     })
    
#     df_exemplo = pd.DataFrame(dados_exemplo)
    
#     # Salva dados de exemplo
#     Path('data').mkdir(exist_ok=True)
#     df_exemplo.to_csv('data/dados_brutos_exemplo.csv', index=False)
#     print(f"Dados de exemplo criados: {len(df_exemplo)} registros")
#     print(f"Colunas: {list(df_exemplo.columns)}")
#     print(f"Valores ausentes: {df_exemplo.isnull().sum().sum()}")
#     print("\n" + "="*60 + "\n")
    
#     # Executa pré-processamento
#     preprocessador = PreprocessadorDadosOrcamentarios()
#     df_processado, relatorio = preprocessador.executar_preprocessamento_completo(
#         caminho_arquivo='data/dados_brutos_exemplo.csv',
#         tipo_arquivo='csv',
#         salvar_processado=True,
#         caminho_saida='data/dados_processados.csv'
#     )
    
#     # Exibe resultados
#     print("\n" + "="*60)
#     print("RESULTADOS DO PRÉ-PROCESSAMENTO")
#     print("="*60)
#     print(f"\n📊 Dados originais: {relatorio['dados_originais']['linhas']} linhas")
#     print(f"📊 Dados processados: {relatorio['dados_processados']['linhas']} linhas")
#     print(f"📉 Linhas removidas: {relatorio['transformacoes']['linhas_removidas']} "
#           f"({relatorio['transformacoes']['percentual_reducao']:.2f}%)")
    
#     print("\n📋 Primeiras linhas dos dados processados:")
#     print(df_processado.head(10))
    
#     print("\n📋 Informações dos dados processados:")
#     print(df_processado.info())
    
#     print("\n📋 Estatísticas descritivas:")
#     print(df_processado.describe())
    
#     return df_processado, relatorio


# # Script principal para execução direta
# if __name__ == "__main__":
#     # Executa demonstração
#     df_final, relatorio_final = demonstrar_preprocessamento()