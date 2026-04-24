import os
import glob
try:
    import pandas as pd
except ImportError:
    print("Pandas is not installed. Please install it using 'pip install pandas'")
    exit(1)

# Defina o diretório onde estão os arquivos
os.chdir("data")

# Lista todos os arquivos .csv
all_filenames = [i for i in glob.glob('*.csv')]

# Concatena e salva
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv.to_csv("arquivo_final.csv", index=False, encoding='utf-8-sig')   