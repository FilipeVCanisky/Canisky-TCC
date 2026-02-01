import numpy as np
from scipy import signal
import cv2
import os

# --- CONFIGURAÇÕES ---
IMG_SIZE = 32
FS = 360  # Frequência de amostragem do MIT-BIH

def converter_para_2d(X_data, nome_dataset):
    """
    Função que recebe um array de sinais (N, 360, 1) 
    e retorna um array de imagens (N, 32, 32, 1)
    """
    print(f"\n>>> Iniciando Conversão do dataset: {nome_dataset}")
    print(f"Shape de entrada: {X_data.shape}")
    
    X_2d_list = []
    total = len(X_data)
    
    for i in range(total):
        if i % 1000 == 0:
            print(f"Processando {i}/{total}...")

        # 1. Flatten: (360, 1) -> (360,) para o scipy entender
        sig = X_data[i].flatten()
        
        # 2. STFT (Spectrograma)
        f, t, Sxx = signal.spectrogram(sig, fs=FS, nperseg=64, noverlap=32)
        
        # 3. Escala Logarítmica (para realçar detalhes)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        
        # 4. Normalização Min-Max (0 a 1)
        img_min, img_max = Sxx_log.min(), Sxx_log.max()
        if img_max - img_min != 0:
            img_norm = (Sxx_log - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(Sxx_log)
            
        # 5. Resize para 32x32 (compatível com CNN leve)
        img_resized = cv2.resize(img_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
        X_2d_list.append(img_resized)

    # Converter lista para array e adicionar canal
    X_2d_final = np.array(X_2d_list)
    X_2d_final = np.expand_dims(X_2d_final, axis=-1) # (N, 32, 32) -> (N, 32, 32, 1)
    
    return X_2d_final

# --- EXECUÇÃO ---

# 1. Carregar Dados de TREINO
if os.path.exists('X_train.npy'):
    X_train_raw = np.load('X_train.npy')
    # Converter
    X_train_img = converter_para_2d(X_train_raw, "TREINO")
    # Salvar
    np.save('X_train_2d.npy', X_train_img)
    print(f"Sucesso! 'X_train_2d.npy' salvo. Shape: {X_train_img.shape}")
    
    # Limpar memória (opcional, bom para PCs com pouca RAM)
    del X_train_raw, X_train_img
else:
    print("ERRO: 'X_train.npy' não encontrado.")

# 2. Carregar Dados de TESTE
if os.path.exists('X_test.npy'):
    X_test_raw = np.load('X_test.npy')
    # Converter
    X_test_img = converter_para_2d(X_test_raw, "TESTE")
    # Salvar
    np.save('X_test_2d.npy', X_test_img)
    print(f"Sucesso! 'X_test_2d.npy' salvo. Shape: {X_test_img.shape}")
else:
    print("ERRO: 'X_test.npy' não encontrado.")

print("\n--- Processo Finalizado ---")