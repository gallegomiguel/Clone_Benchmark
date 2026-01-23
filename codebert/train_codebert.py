import pandas as pd
import torch
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import random

BATCH_SIZE = 16 
EPOCHS = 3      
MAX_LEN = 256   
LR = 2e-5       

DATA_PATH = "./codebert_train_10.csv"
MODEL_SAVE_PATH = "./codebert_finetuned"

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def prepare_data(df, tokenizer):
    print("   Tokenizando datos... (Esto puede tardar un poco)")
    inputs = tokenizer(
        df['code1'].tolist(),
        df['code2'].tolist(),
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    labels = torch.tensor(df['label'].tolist())
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    return dataset

def train():
    print(f"--- INICIANDO CODEBERT (Batch: {BATCH_SIZE}, Len: {MAX_LEN}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Dispositivo: {device}")
    
    print("1. Cargando datos...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: No encuentro el archivo en {DATA_PATH}")
        return

    df = df.dropna().reset_index(drop=True)
    print(f"   Total filas: {len(df)}")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"   Train: {len(train_df)} | Test: {len(test_df)}")

    print("2. Cargando Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    train_dataset = prepare_data(train_df, tokenizer)
    test_dataset = prepare_data(test_df, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    print("3. Cargando Modelo Pre-entrenado...")
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    
    print(f"\n4. ¡ENTRENANDO! ({EPOCHS} Épocas)")
    total_start_time = time.time()
    
    for epoch_i in range(0, EPOCHS):
        print(f'\n======== Época {epoch_i + 1} / {EPOCHS} ========')
        t0 = time.time()
        total_loss = 0
        model.train()

        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)
        training_time = time.time() - t0
        
        print(f"  [CodeBERT] Epoch {epoch_i + 1} | Time: {training_time:.2f}s | Avg Loss: {avg_train_loss:.4f}")

    print(f"\n=== ENTRENAMIENTO FINALIZADO. Tiempo Total: {time.time() - total_start_time:.2f}s ===")

    print("\n5. EJECUTANDO PREDICCIÓN EN TEST SET...")
    model.eval()
    
    predictions , true_labels = [], []
    
    t0_test = time.time()
    for batch in tqdm(test_dataloader, desc="Testing"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        preds = np.argmax(logits, axis=1).flatten()
        predictions.extend(preds)
        true_labels.extend(label_ids)
    
    test_time = time.time() - t0_test
    avg_inference_time = test_time / len(test_dataloader.dataset)
    
    print(f"   Tiempo de Inferencia Total: {test_time:.2f}s")
    print(f"   Tiempo por muestra: {avg_inference_time:.6f}s")

    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    import json
    metrics = {
        "model": "CodeBERT",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "avg_inference_time": avg_inference_time
    }
    print(f"__DATA_JSON__{json.dumps(metrics)}")

if __name__ == '__main__':
    train()