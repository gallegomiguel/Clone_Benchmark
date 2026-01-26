import pandas as pd
import os
from tqdm import tqdm

base_path = "../astnn/data/java"

input_astnn_pkl = f"{base_path}/train/blocks.pkl"
input_raw_code  = f"{base_path}/bcb_funcs_all.tsv"
output_astnn_pkl = f"{base_path}/train/blocks_10percent.pkl"
output_codebert_csv = "../codebert/codebert_train_10.csv"

SAMPLE_RATE = 0.10

print("--- INICIANDO REDUCCIÓN CONJUNTA (VERSIÓN ROBUSTA) ---")

print(f"1. Cargando pares originales...")
try:
    df_full = pd.read_pickle(input_astnn_pkl)
    print(f"   Total pares originales: {len(df_full)}")
except FileNotFoundError:
    print("ERROR: No encuentro blocks.pkl. Verifica la ruta.")
    exit()

def stratified_sample(group):
    if len(group) < 50: return group
    return group.sample(frac=SAMPLE_RATE, random_state=42)

print("2. Aplicando reducción al 10%...")
df_reduced = df_full.groupby('label', group_keys=False).apply(stratified_sample)
print(f"   Pares finales: {len(df_reduced)}")

df_reduced.to_pickle(output_astnn_pkl)
print("   Archivo ASTNN guardado.")

print("3. Cargando código fuente...")
code_map = {}
current_id = None
current_code_buffer = []

with open(input_raw_code, 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, desc="Leyendo TSV"):
        parts = line.split('\t', 1)
        try:
            new_id = int(parts[0])
            if current_id is not None:
                code_map[current_id] = "".join(current_code_buffer)
            current_id = new_id
            current_code_buffer = []
            if len(parts) > 1:
                current_code_buffer.append(parts[1])

        except ValueError:  # si esta línea es continuación del código anterior
            if current_id is not None:
                current_code_buffer.append(line)

    if current_id is not None:
        code_map[current_id] = "".join(current_code_buffer)

print(f"   -> Código cargado correctamente: {len(code_map)} funciones.")

print("4. Cruzando datos para CodeBERT...")
codebert_data = []
fails = 0

for index, row in tqdm(df_reduced.iterrows(), total=len(df_reduced), desc="Generando CSV"):
    id1, id2, label = int(row['id1']), int(row['id2']), int(row['label'])
    c1 = code_map.get(id1)
    c2 = code_map.get(id2)

    if c1 and c2:
        codebert_data.append({
            'code1': c1,
            'code2': c2,
            'label': 1 if label > 0 else 0,
            'original_type': label
        })
    else:
        fails += 1

print(f"   (Pares ignorados por no encontrar código: {fails})")
df_codebert = pd.DataFrame(codebert_data)
df_codebert.to_csv(output_codebert_csv, index=False)
print(f"   Archivo CodeBERT guardado en: {output_codebert_csv}")
print("¡LISTO! Ahora sí.")