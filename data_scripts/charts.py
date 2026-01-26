import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ORIGINAL_PATH = "../astnn/clone/data/java/train/blocks.pkl"
REDUCED_PATH = "../astnn/clone/data/java/train/blocks_10percent.pkl"
OUTPUT_IMG = "distribucion_clases.png"

LABEL_MAP = {
    0: 'No Clon',
    1: 'Type-1',
    2: 'Type-2',
    3: 'Type-3',
    4: 'Type-4',
    5: 'Type-5'
}

def get_distribution(path, name):
    print(f"--- Analizando {name} ---")
    if not os.path.exists(path):
        print(f"⚠️ Error: No encuentro {path}")
        return None

    try:
        df = pd.read_pickle(path)
        print(f"   Total filas: {len(df)}")

        counts = df['label'].value_counts(normalize=True).sort_index()  # porcentajes
        return counts
    except Exception as e:
        print(f"   Error leyendo pickle: {e}")
        return None


dist_orig = get_distribution(ORIGINAL_PATH, "Dataset ORIGINAL")
dist_red = get_distribution(REDUCED_PATH, "Dataset REDUCIDO (10%)")

if dist_orig is not None and dist_red is not None:
    print("\nGenerando gráfica comparativa...")

    all_labels = sorted(list(set(dist_orig.index) | set(dist_red.index)))
    labels_str = [LABEL_MAP.get(l, str(l)) for l in all_labels]

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    vals_orig = [dist_orig.get(l, 0) * 100 for l in all_labels]
    rects1 = ax.bar(x - width/2, vals_orig, width, label='Original (100%)', color='#A0CBE8')
    vals_red = [dist_red.get(l, 0) * 100 for l in all_labels]
    rects2 = ax.bar(x + width/2, vals_red, width, label='Reducido (10%)', color='#FFBE7D')

    ax.set_ylabel('Porcentaje del Dataset (%)')
    ax.set_title('Distribución de Tipos de Clones: Original vs Muestra')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_str)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Gráfica guardada como: {OUTPUT_IMG}")
    plt.show()