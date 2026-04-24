import os
import glob
import pandas as pd

os.chdir("data")

all_filenames = [i for i in glob.glob('*.csv')]

# Concatena e salva
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv.to_csv("arquivo_final.csv", index=False, encoding='utf-8-sig')   