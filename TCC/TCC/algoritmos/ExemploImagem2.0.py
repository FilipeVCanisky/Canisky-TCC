import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import os

# --- 1. CARREGAMENTO DOS DADOS ---
print("Carregando amostras...")

if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
    X_source = np.load('X_test.npy')
    y_source = np.load('y_test.npy')
    print(f"Fonte: TESTE (Original). Shape: {X_source.shape}")
elif os.path.exists('X_train.npy'):
    print("Aviso: Usando TREINO (X_train.npy).")
    X_source = np.load('X_train.npy')
    y_source = np.load('y_train.npy')
else:
    print("ERRO: Nenhum arquivo .npy encontrado.")
    exit()

# --- 2. CONFIGURAÇÕES ---
classes_map = {
    0: 'N (Normal)',
    1: 'S (Supraventricular)',
    2: 'V (Ventricular)',
    3: 'F (Fusão)',
    4: 'Q (Desconhecido)'
}

# Nomes curtos para o arquivo
classes_sigla = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

FS = 360
IMG_SIZE = 32

def get_spectrogram_visual(sig_1d):
    f, t, Sxx = signal.spectrogram(sig_1d, fs=FS, nperseg=64, noverlap=32)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    img_min, img_max = Sxx_log.min(), Sxx_log.max()
    if img_max - img_min != 0:
        img_norm = (Sxx_log - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(Sxx_log)
    img_resized = cv2.resize(img_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img_resized

# Cria pasta para salvar
pasta_saida = "imagens_individuais"
os.makedirs(pasta_saida, exist_ok=True)

print(f"\nGerando imagens individuais em: '{pasta_saida}/' ...")

# --- 3. LOOP POR CLASSE ---
for class_id in range(5):
    indices = np.where(y_source == class_id)[0]
    
    if len(indices) > 0:
        # Pega um exemplo aleatório
        idx = indices[np.random.randint(0, len(indices))]
        original_sig = X_source[idx].flatten()
        img_2d = get_spectrogram_visual(original_sig)
        
        # --- CRIA UMA FIGURA NOVA PARA CADA CLASSE ---
        # 1 linha, 2 colunas
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot 1: Sinal no Tempo
        ax1.plot(original_sig, color='black', linewidth=1.5)
        ax1.set_title(f"Sinal Original - {classes_map[class_id]}")
        ax1.set_xlabel("Amostras")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 360)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Espectrograma
        ax2.imshow(img_2d, cmap='gray', origin='lower', aspect='auto')
        ax2.set_title("Espectrograma (Entrada CNN)")
        ax2.set_xlabel("Janelas de Tempo")
        ax2.set_ylabel("Frequência")
        
        # Ajuste fino
        plt.tight_layout()
        
        # Salvar arquivo individual
        nome_arquivo = f"Exemplo_Classe_{classes_sigla[class_id]}.png"
        caminho_completo = os.path.join(pasta_saida, nome_arquivo)
        
        plt.savefig(caminho_completo, dpi=300)
        print(f"Salvo: {nome_arquivo}")
        
        # Fecha a figura para liberar memória
        plt.close(fig)

    else:
        print(f"Aviso: Classe {class_id} não tem amostras.")

print("\nProcesso concluído!")