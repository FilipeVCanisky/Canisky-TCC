import numpy as np
import wfdb
from sklearn.utils import resample
from collections import Counter

# Configurações
OUTPUT_SIZE = 360   # tamanho da janela
SHIFT_LIMIT = 20    # deslocamento para aumento de dados
INPUT_WINDOW = OUTPUT_SIZE + (2 * SHIFT_LIMIT) 
HALF_WINDOW = INPUT_WINDOW // 2

# quantidade de casos
TARGET_TRAIN = 1600
TARGET_EVAL_FIXED = 200 # Usado tanto para Teste quanto para Validação (200 cada)

# ecg disponíveis
RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASS_INT_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

X_raw = [] 
y_raw = [] 

print(f"Iniciando extração")

# Leitura e extração
for rec in RECORDS:
    try:
        signals, fields = wfdb.rdsamp(rec, channels=[0], pn_dir='mitdb')
        signal_array = signals.flatten()
        ann = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
        
        for i, symbol in enumerate(ann.symbol):
            if symbol in AAMI_MAP:
                peak_index = ann.sample[i]
                
                if (peak_index - HALF_WINDOW < 0) or (peak_index + HALF_WINDOW > len(signal_array)):
                    continue
                
                segment = signal_array[peak_index - HALF_WINDOW : peak_index + HALF_WINDOW]
                
                if np.max(segment) != np.min(segment):
                    segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
                else:
                    segment = np.zeros_like(segment)

                aami_label = AAMI_MAP[symbol]
                X_raw.append(segment)
                y_raw.append(CLASS_INT_MAP[aami_label])
                
    except Exception as e:
        print(f"Erro ao ler registro {rec}: {e}")

X = np.array(X_raw)
y = np.array(y_raw)
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Dados totais extraídos: {dict(Counter(y))}")

# divisão
print("\nSeparando Treino, Validação e Teste (Priorizando 200 Val / 200 Teste)")

X_train_ext = []
y_train = []
X_val_ext = []
y_val = []
X_test_ext = []
y_test = []

for class_id in range(5):
    # Pega todos os dados dessa classe
    ids = np.where(y == class_id)[0]
    X_class_all = X[ids]
    y_class_all = y[ids]
    
    total_class = len(X_class_all)
    
    # Embaralha para garantir aleatoriedade
    X_class_all, y_class_all = resample(X_class_all, y_class_all, replace=False, random_state=42)
    
    # Corte fixo pois a menor classe tem 800 casos
    cutoff = TARGET_EVAL_FIXED 
        
    print(f"Classe {class_id} (Total: {total_class}): {cutoff} Teste | {cutoff} Validação | {total_class - (2*cutoff)} Treino Base")

    # Fatia os dados
    # 1. TESTE: Pega os primeiros cutoff
    X_test_ext.extend(X_class_all[:cutoff])
    y_test.extend(y_class_all[:cutoff])
    
    # 2. VALIDAÇÃO: Pega a segunda fatia de cutoff
    val_end = cutoff * 2
    X_val_ext.extend(X_class_all[cutoff:val_end])
    y_val.extend(y_class_all[cutoff:val_end])
    
    # 3. TREINO: Pega todo o resto que sobrou
    X_train_ext.extend(X_class_all[val_end:])
    y_train.extend(y_class_all[val_end:])

X_train_ext = np.array(X_train_ext)
y_train = np.array(y_train)
X_val_ext = np.array(X_val_ext)
y_val = np.array(y_val)
X_test_ext = np.array(X_test_ext)
y_test = np.array(y_test)

# recorte
def process_signal(signal_extended, augment=False):
    length = len(signal_extended) 
    crop_size = OUTPUT_SIZE       
    
    if not augment:
        start_index = (length - crop_size) // 2
        return signal_extended[start_index : start_index + crop_size]

    choice = np.random.randint(0, 3) 
    
    if choice == 0: # SHIFT
        max_start = length - crop_size
        start_index = np.random.randint(0, max_start + 1)
        return signal_extended[start_index : start_index + crop_size]

    start_index_center = (length - crop_size) // 2
    signal_centered = signal_extended[start_index_center : start_index_center + crop_size]
    
    if choice == 1: # RUÍDO
        noise = np.random.normal(0, 0.01, size=signal_centered.shape)
        return signal_centered + noise
        
    elif choice == 2: # ESCALA
        factor = np.random.uniform(0.95, 1.05)
        return signal_centered * factor

print("\nBalanceando treino (Data Augmentation)...")

X_train_final = []
y_train_final = []

for class_id in range(5):
    ids = np.where(y_train == class_id)[0]
    X_class = X_train_ext[ids]
    y_class = y_train[ids]
    n_samples = len(X_class)
    
    if n_samples > TARGET_TRAIN:
        # Downsampling
        X_res, y_res = resample(X_class, y_class, replace=False, n_samples=TARGET_TRAIN, random_state=42)
        for sig in X_res:
            X_train_final.append(process_signal(sig, augment=False))
            y_train_final.append(class_id)
            
    else:
        # 1. Guarda os originais que sobraram
        for sig in X_class:
            X_train_final.append(process_signal(sig, augment=False))
            y_train_final.append(class_id)
            
        # 2. Gera novos até completar 1600
        needed = TARGET_TRAIN - n_samples
        if n_samples > 0:
            indices_to_clone = np.random.randint(0, n_samples, needed)
            for idx in indices_to_clone:
                raw_sig = X_class[idx]
                X_train_final.append(process_signal(raw_sig, augment=True))
                y_train_final.append(class_id)

X_train_final = np.array(X_train_final)
y_train_final = np.array(y_train_final)
X_train_final, y_train_final = resample(X_train_final, y_train_final, replace=False, random_state=42)

print("\nProcessando validação...")
X_val_final = []
y_val_final = y_val 

for sig in X_val_ext:
    # Apenas recorta no centro
    X_val_final.append(process_signal(sig, augment=False))

X_val_final = np.array(X_val_final)

print("\nProcessando teste...")
X_test_final = []
y_test_final = y_test 

for sig in X_test_ext:
    # Apenas recorta no centro
    X_test_final.append(process_signal(sig, augment=False))

X_test_final = np.array(X_test_final)

# salvando pra n ter q rodar toda vez
print(f"SHAPE FINAL TREINO:    {X_train_final.shape}")
print(f"Distribuição Treino:    {dict(Counter(y_train_final))}")
print(f"SHAPE FINAL VALIDAÇÃO: {X_val_final.shape}")
print(f"Distribuição Validação: {dict(Counter(y_val_final))}")
print(f"SHAPE FINAL TESTE:     {X_test_final.shape}")
print(f"Distribuição Teste:     {dict(Counter(y_test_final))}")

np.save('X_train.npy', X_train_final)
np.save('y_train.npy', y_train_final)
np.save('X_val.npy', X_val_final)
np.save('y_val.npy', y_val_final)
np.save('X_test.npy', X_test_final)
np.save('y_test.npy', y_test_final)

print("\nArquivos salvos com sucesso!")