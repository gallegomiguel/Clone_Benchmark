import subprocess
import json
import numpy as np
import os
import sys

# --- CONFIGURACIÓN ---
NUM_RUNS = 5  # Número de veces que repetiremos el experimento
PYTHON_EXE = sys.executable # Usa el mismo python que está ejecutando esto

# Listas para guardar resultados
results = {
    "ASTNN": {"f1": [], "precision": [], "recall": []},
    "CodeBERT": {"f1": [], "precision": [], "recall": []}
}

def run_script(script_path, work_dir, model_name):
    print(f"🚀 Ejecutando {model_name} (En: {work_dir})...")
    
    # Lanzamos el proceso
    # cwd=work_dir es CRÍTICO: hace que el script crea que está en su carpeta
    process = subprocess.Popen(
        [PYTHON_EXE, script_path, "--lang", "java"], # Argumentos para ASTNN (CodeBERT los ignora)
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Capturamos errores también
        text=True
    )
    
    # Leemos la salida línea a línea mientras se ejecuta
    output_json = None
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"   [{model_name}] {line.strip()}") # Opcional: ver logs en tiempo real
            if "__DATA_JSON__" in line:
                # ¡Bingo! Encontramos la línea con los datos
                json_str = line.split("__DATA_JSON__")[1].strip()
                output_json = json.loads(json_str)

    if output_json:
        print(f"✅ Resultados capturados: F1={output_json['f1']:.4f}")
        return output_json
    else:
        print(f"❌ Error: No se encontraron métricas finales para {model_name}")
        # Imprimir errores si falló
        print(process.stderr.read())
        return None

# --- BUCLE PRINCIPAL ---
print(f"--- INICIANDO BENCHMARK ({NUM_RUNS} EJECUCIONES) ---\n")

for i in range(1, NUM_RUNS + 1):
    print(f"\n🔁 === RONDA {i}/{NUM_RUNS} ===")
    
    # 1. Ejecutar ASTNN
    # Asumimos estructura: ./astnn/train.py
    astnn_metrics = run_script("train.py", "astnn", "ASTNN") # Ojo a la ruta interna de ASTNN
    if astnn_metrics:
        results["ASTNN"]["f1"].append(astnn_metrics["f1"])
        results["ASTNN"]["precision"].append(astnn_metrics["precision"])
        results["ASTNN"]["recall"].append(astnn_metrics["recall"])

    # 2. Ejecutar CodeBERT
    # Asumimos estructura: ./codebert/train_codebert.py
    cb_metrics = run_script("train_codebert.py", "codebert", "CodeBERT")
    if cb_metrics:
        results["CodeBERT"]["f1"].append(cb_metrics["f1"])
        results["CodeBERT"]["precision"].append(cb_metrics["precision"])
        results["CodeBERT"]["recall"].append(cb_metrics["recall"])

# --- INFORME FINAL ---
print("\n\n📊 === INFORME FINAL DE RESULTADOS ===")

for model in ["ASTNN", "CodeBERT"]:
    print(f"\n🔹 Modelo: {model}")
    if len(results[model]["f1"]) > 0:
        mean_f1 = np.mean(results[model]["f1"])
        std_f1 = np.std(results[model]["f1"])
        mean_p = np.mean(results[model]["precision"])
        mean_r = np.mean(results[model]["recall"])
        
        print(f"   Ejecuciones exitosas: {len(results[model]['f1'])}")
        print(f"   F1-Score  : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"   Precision : {mean_p:.4f}")
        print(f"   Recall    : {mean_r:.4f}")
    else:
        print("   ⚠️ No hay datos (¿Fallaron todas las ejecuciones?)")

# Guardar en fichero
with open("resultados_benchmarking.txt", "w") as f:
    f.write(str(results))