import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import random
import pandas as pd

from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

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
        
        # Evitar divisão por zero
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
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
    media_geral = df.mean(numeric_only=True)
    df.loc['Média'] = media_geral
    df.at['Média', 'Classe'] = 'MÉDIA GERAL'
    
    return df

# condigurações
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

files_needed = ['X_train_2d.npy', 'y_train.npy', 'X_test_2d.npy', 'y_test.npy']
for f in files_needed:
    if not os.path.exists(f):
        print(f"ERRO CRÍTICO: Arquivo {f} não encontrado. Rode os scripts anteriores.")
        exit()

print("Carregando")
X_train = np.load('X_train_2d.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test_2d.npy')
y_test = np.load('y_test.npy')

print(f"Shape Treino: X={X_train.shape}, y={y_train.shape}")
print(f"Shape Teste:  X={X_test.shape}, y={y_test.shape}")

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

n_canais = X_train.shape[-1] # Deve ser 1

# arquitetura
model = Sequential()

# Bloco 1
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, n_canais)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Bloco 2
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Bloco 3
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Classificador
model.add(Flatten())
model.add(Dropout(0.5)) 
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Otimizador
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

# callbacks
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

print("\nTreinamento")

history = model.fit(
    X_train, y_train_cat,
    epochs=200, 
    batch_size=64,
    validation_data=(X_test, y_test_cat),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

print("Treinamento concluído!")

# avaliação

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

y_true = y_test 

classes_nomes = ['N', 'S', 'V', 'F', 'Q']

tabela_resultados = gerar_tabela_metricas(y_true, y_pred, classes_nomes)

print("\nTABELA FINAL DE RESULTADOS")
print(tabela_resultados.to_string(index=False))
