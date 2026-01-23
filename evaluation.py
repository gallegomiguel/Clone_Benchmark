import subprocess
import json
import numpy as np
import os
import sys

NUM_RUNS = 15  # CAMBIAR: Número de veces que repetiremos el experimento

PYTHON_EXE = sys.executable
results = {
    "ASTNN": {"f1": [], "precision": [], "recall": [], "time": []}, # Añadido "time"
    "CodeBERT": {"f1": [], "precision": [], "recall": [], "time": []} # Añadido "time"
}

def run_script(script_path, work_dir, model_name):
    print(f"🚀 Ejecutando {model_name} (En: {work_dir})...")
    process = subprocess.Popen(
        [PYTHON_EXE, script_path, "--lang", "java"],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    output_json = None
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"   [{model_name}] {line.strip()}")
            if "__DATA_JSON__" in line:  # Encontramos la línea con los datos
                json_str = line.split("__DATA_JSON__")[1].strip()
                output_json = json.loads(json_str)

    if output_json:
        print(f"Resultados: F1={output_json['f1']:.4f}")
        return output_json
    else:
        print(f"Error: No se encontraron métricas finales para {model_name}")
        print(process.stderr.read())
        return None

# --- BUCLE PRINCIPAL ---
print(f"--- INICIANDO BENCHMARK ({NUM_RUNS} EJECUCIONES) ---\n")

for i in range(1, NUM_RUNS + 1):
    print(f"\n=== VUELTA {i}/{NUM_RUNS} ===")
    
    # 1. ASTNN
    astnn_metrics = run_script("train.py", "astnn", "ASTNN")
    if astnn_metrics:
        results["ASTNN"]["f1"].append(astnn_metrics["f1"])
        results["ASTNN"]["precision"].append(astnn_metrics["precision"])
        results["ASTNN"]["recall"].append(astnn_metrics["recall"])
        results["ASTNN"]["time"].append(astnn_metrics["avg_inference_time"])

    # 2. CodeBERT
    cb_metrics = run_script("train_codebert.py", "codebert", "CodeBERT")
    if cb_metrics:
        results["CodeBERT"]["f1"].append(cb_metrics["f1"])
        results["CodeBERT"]["precision"].append(cb_metrics["precision"])
        results["CodeBERT"]["recall"].append(cb_metrics["recall"])
        results["CodeBERT"]["time"].append(cb_metrics["avg_inference_time"])

print("\n\n=== INFORME FINAL DE RESULTADOS ===")

for model in ["ASTNN", "CodeBERT"]:
    print(f"\n🔹 Modelo: {model}")
    if len(results[model]["f1"]) > 0:
        mean_f1 = np.mean(results[model]["f1"])
        std_f1 = np.std(results[model]["f1"])
        mean_time = np.mean(results[model]["time"]) # Segundos
        std_time = np.std(results[model]["time"])
        
        print(f"   F1-Score      : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"   Inferencia (s): {mean_time:.6f}s ± {std_time:.6f}s") # 6 decimales porque es muy rápido
        print(f"   Inferencia (ms): {mean_time * 1000:.2f}ms") # Dato más legible para humanos
    else:
        print("No hay datos disponibles.")

with open("resultados_benchmarking.txt", "w") as f:
    f.write(str(results))