import pandas as pd
import os
from tqdm import tqdm

# --- 1. CONFIGURACIÓN DE RUTAS ---
base_path = "../astnn/data/java"

# Archivos de entrada
input_astnn_pkl = f"{base_path}/train/blocks.pkl"
input_raw_code  = f"{base_path}/bcb_funcs_all.tsv"

# Archivos de salida
output_astnn_pkl = f"{base_path}/train/blocks_10percent.pkl"
output_codebert_csv = "../codebert/codebert_train_10.csv"

SAMPLE_RATE = 0.10  # 10%

print("--- INICIANDO REDUCCIÓN CONJUNTA (VERSIÓN ROBUSTA) ---")

# --- 2. CARGAR Y REDUCIR PARES ---
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
# El warning que te salió antes es inofensivo, pero esto lo silencia o lo evita
df_reduced = df_full.groupby('label', group_keys=False).apply(stratified_sample)
print(f"   Pares finales: {len(df_reduced)}")

# Guardamos el de ASTNN
df_reduced.to_pickle(output_astnn_pkl)
print("   ✅ Archivo ASTNN guardado.")

# --- 3. CARGAR CÓDIGO RAW (LA PARTE QUE FALLABA) ---
print("3. Cargando código fuente (Modo Inteligente)...")
code_map = {}

current_id = None
current_code_buffer = []

with open(input_raw_code, 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, desc="Leyendo TSV"):
        parts = line.split('\t', 1)

        # Intentamos ver si la línea empieza por un ID numérico
        try:
            # Si el primer trozo es un número, ¡es una función nueva!
            new_id = int(parts[0])

            # Guardamos la función ANTERIOR antes de empezar la nueva
            if current_id is not None:
                code_map[current_id] = "".join(current_code_buffer)

            # Reseteamos para la nueva función
            current_id = new_id
            current_code_buffer = []
            if len(parts) > 1:
                current_code_buffer.append(parts[1]) # Guardamos el resto de la línea

        except ValueError:
            # Si falla el int(), es que esta línea es continuación del código anterior
            # Simplemente la añadimos al buffer
            if current_id is not None:
                current_code_buffer.append(line)

    # Importante: Guardar la ultimísima función al acabar el bucle
    if current_id is not None:
        code_map[current_id] = "".join(current_code_buffer)

print(f"   -> Código cargado correctamente: {len(code_map)} funciones.")

# --- 4. CRUZAR DATOS ---
print("4. Cruzando datos para CodeBERT...")
codebert_data = []

# Convertimos a diccionario para búsqueda rápida (ya lo es, pero por si acaso)
# Nota: df_reduced tiene id1 e id2.
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
print(f"   ✅ Archivo CodeBERT guardado en: {output_codebert_csv}")
print("¡LISTO! Ahora sí.")