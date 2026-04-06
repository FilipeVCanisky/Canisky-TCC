import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import os

print("Carregando amostras para visualização...")

# Prioriza o X_test para pegar dados REAIS
if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
    X_source = np.load('X_test.npy')
    y_source = np.load('y_test.npy')
    print(f"Fonte de dados: CONJUNTO DE TESTE (Original). Shape: {X_source.shape}")
elif os.path.exists('X_train.npy'):
    print("Aviso: 'X_test.npy' não encontrado. Usando 'X_train.npy'.")
    X_source = np.load('X_train.npy')
    y_source = np.load('y_train.npy')
else:
    print("ERRO CRÍTICO: Nenhum arquivo de dados (.npy) encontrado.")
    exit()

classes_map = {
    0: 'N (Normal)',
    1: 'S (Supraventricular)',
    2: 'V (Ventricular)',
    3: 'F (Fusão)',
    4: 'Q (Desconhecido)'
}

FS = 360
IMG_SIZE = 32

def get_spectrogram_visual(sig_1d):
    # STFT
    f, t, Sxx = signal.spectrogram(sig_1d, fs=FS, nperseg=64, noverlap=32)
    # Log Scale
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    # Normalize
    img_min, img_max = Sxx_log.min(), Sxx_log.max()
    if img_max - img_min != 0:
        img_norm = (Sxx_log - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(Sxx_log)
    # Resize
    img_resized = cv2.resize(img_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img_resized

fig, axes = plt.subplots(5, 2, figsize=(10, 12)) 
plt.subplots_adjust(hspace=0.6, wspace=0.3)

print("\nGerando painel de classes...")

for class_id in range(5):
    indices = np.where(y_source == class_id)[0]
    
    if len(indices) > 0:
        # Sorteia um exemplo
        idx = indices[np.random.randint(0, len(indices))]
        
        original_sig = X_source[idx].flatten()
        img_2d = get_spectrogram_visual(original_sig)
        
        ax1 = axes[class_id, 0]
        ax1.plot(original_sig, color='black', linewidth=1.2) # Preto simples
        ax1.set_title(f"Sinal {classes_map[class_id]}", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Amplitude Norm.")
        ax1.set_xlim(0, 360)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        if class_id < 4:
            ax1.set_xticklabels([])
        else:
            ax1.set_xlabel("Amostras (Tempo)")

        ax2 = axes[class_id, 1]

        ax2.imshow(img_2d, cmap='gray', origin='lower', aspect='auto')
        
        ax2.set_title(f"Espectrograma (CNN Input)", fontsize=11, fontweight='bold')
        ax2.axis('off') 
        
    else:
        axes[class_id, 0].text(0.5, 0.5, "Sem amostras", ha='center')
        axes[class_id, 1].axis('off')
        print(f"Aviso: Nenhum exemplo encontrado para classe {class_id}")

plt.suptitle("Representação 1D vs 2D dos Batimentos Cardíacos", fontsize=16, y=0.95)

plt.savefig("figura_exemplos_tcc_gray.png", dpi=300, bbox_inches='tight')
print("\nFigura salva como 'figura_exemplos_tcc_gray.png'")

plt.show()