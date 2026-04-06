import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import random
import pandas as pd

from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# CONFIGURAÇÕES
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# métricas
def gerar_tabela_metricas(y_true, y_pred, classes_map):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_map, 
                yticklabels=classes_map)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()
    
    metrics_list = []
    
    for i, class_name in enumerate(classes_map):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics_list.append({
            "Classe": class_name,
            "Acurácia": round(accuracy, 4),
            "Sensibilidade": round(sensitivity, 4),
            "Especificidade": round(specificity, 4),
            "Precisão": round(precision, 4),
            "F1-Score": round(f1, 4)
        })
    
    df = pd.DataFrame(metrics_list)
    media_geral = df.mean(numeric_only=True).round(4)
    df.loc['Média'] = media_geral
    df.at['Média', 'Classe'] = 'MÉDIA GERAL'
    
    return df

def plotar_curvas_aprendizado(history):
    plt.figure(figsize=(12, 5))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validação', color='orange')
    plt.title('Acurácia ao longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino', color='blue')
    plt.plot(history.history['val_loss'], label='Validação', color='orange')
    plt.title('Erro (Loss) ao longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# carregamento de dados
print("Carregando arquivos pré-processados")

if not os.path.exists('X_train.npy') or not os.path.exists('X_val.npy'):
    print("ERRO: Arquivos .npy não encontrados. Rode o script de pré-processamento primeiro.")
    exit()

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(f"Treino:    {X_train.shape} - Labels: {y_train.shape}")
print(f"Validação: {X_val.shape}   - Labels: {y_val.shape}")
print(f"Teste:     {X_test.shape}   - Labels: {y_test.shape}")

# One-Hot Encoding
y_train_cat = to_categorical(y_train, num_classes=5)
y_val_cat = to_categorical(y_val, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# modelo
model = Sequential()

# Camada 1
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(360, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Camada 2
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Camada 3
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Classificador
model.add(Flatten())
model.add(Dropout(0.5)) 
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Otimizador
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# callbacks
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

checkpoint = ModelCheckpoint(
    'melhor_modelo_ecg.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1
)

print("\nIniciando treinamento")

history = model.fit(
    X_train, y_train_cat,
    epochs=100, 
    batch_size=64,
    validation_data=(X_val, y_val_cat), 
    callbacks=[lr_scheduler, early_stop, checkpoint],
    verbose=1
)

print("Treinamento concluído!")

plotar_curvas_aprendizado(history)

# Avaliação
print("\nGerando relatórios com o MELHOR modelo salvo...")

# melhor peso salvo
best_model = load_model('melhor_modelo_ecg.keras')

# A avaliação final é feita estritamente sobre o conjunto de teste inédito
y_pred_prob = best_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = y_test 

classes_nomes = ['N', 'S', 'V', 'F', 'Q']

tabela_resultados = gerar_tabela_metricas(y_true, y_pred, classes_nomes)

print("\nTABELA FINAL DE RESULTADOS")
print(tabela_resultados.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, float) else x))