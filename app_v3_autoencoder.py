"""
================================================================================
PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS - FASE 2: AUTOENCODER
================================================================================
Disciplina: Visão por Computador
Dataset: MVTec AD (bottle)
Objetivo: Detectar anomalias usando Aprendizado Não-Supervisionado

CONCEITO (AUTOENCODER):
Diferente da Fase 1 (CNN), aqui vamos treinar a rede APENAS com imagens normais.
A rede aprenderá a "comprimir" a imagem boa num vetor pequeno (Encoder) e 
"descomprimir" de volta para a imagem original (Decoder).
Quando passarmos uma imagem DEFEITUOSA pela rede, como ela só sabe reconstruir 
garrafas normais, a reconstrução sairá com defeito (ou seja, ela vai tentar desenhar 
uma garrafa normal por cima do defeito). A diferença matemática entre a imagem 
original e a reconstruída nos dará um mapa de calor exato de onde está a anomalia!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# Suprimir warnings
tf.get_logger().setLevel('ERROR')

print("=" * 80)
print("FASE 2: DETECÇÃO DE ANOMALIAS COM AUTOENCODER")
print("=" * 80)

# ==============================================================================
# 1. CONFIGURAÇÕES E CARREGAMENTO DOS DADOS
# ==============================================================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 30 # Autoencoders costumam precisar de mais épocas para convergir

BASE_PATH = "/home/mafonso/Documents/Mestrado UPT/Visão por computador/UPT_VPC/industrial_defect_project_pro/dataset/bottle"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

def load_images(folder_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    if not os.path.exists(folder_path): return np.array([])
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).resize(img_size)
            img = np.array(img) / 255.0  # Normalização [0, 1]
            images.append(img)
    return np.array(images)

print("\nCarregando dados (APENAS IMAGENS NORMAIS PARA TREINO)...")
# O grande truque: O treino só vê imagens sem defeito!
X_train = load_images(os.path.join(TRAIN_PATH, "good"))
print(f"✅ Treino: {len(X_train)} imagens normais")

print("\nCarregando dados de teste (Normais e Defeituosas)...")
X_test_good = load_images(os.path.join(TEST_PATH, "good"))
X_test_broken_large = load_images(os.path.join(TEST_PATH, "broken_large"))
X_test_broken_small = load_images(os.path.join(TEST_PATH, "broken_small"))
X_test_contamination = load_images(os.path.join(TEST_PATH, "contamination"))

# Criando conjunto de teste completo e seus rótulos (0=Normal, 1=Anomalia)
X_test = np.concatenate([X_test_good, X_test_broken_large, X_test_broken_small, X_test_contamination])
y_test = np.concatenate([np.zeros(len(X_test_good)), 
                         np.ones(len(X_test_broken_large) + len(X_test_broken_small) + len(X_test_contamination))])

print(f"✅ Teste Total: {len(X_test)} imagens ({len(X_test_good)} Normais, {len(X_test) - len(X_test_good)} Defeitos)")

# ==============================================================================
# 2. ARQUITETURA DO AUTOENCODER
# ==============================================================================

print("\n" + "=" * 80)
print("CONSTRUINDO O AUTOENCODER")
print("=" * 80)

input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# ENCODER (Comprime a informação)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x) # 64x64
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x) # 32x32
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x) # 16x16x8 (Representação latente/comprimida)

# DECODER (Reconstrói a informação)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x) # 32x32
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x) # 64x64
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x) # 128x128
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # Imagem final

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error como função de perda
autoencoder.summary()

# ==============================================================================
# 3. TREINAMENTO
# ==============================================================================

print("\nTreinando o Autoencoder para RECONSTRUIR imagens normais...")
# O input e o target são OS MESMOS (X_train, X_train)!
history = autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# ==============================================================================
# 4. AVALIAÇÃO E CÁLCULO DO ERRO DE RECONSTRUÇÃO (MAPAS DE CALOR)
# ==============================================================================

print("\n" + "=" * 80)
print("CRIANDO MAPAS DE CALOR DE ANOMALIAS")
print("=" * 80)

# O Autoencoder tenta reconstruir as imagens de teste
reconstructed = autoencoder.predict(X_test)

# O "erro" é a diferença absoluta entre a imagem original e a reconstruída
# Tiramos a média dos canais RGB para ter um mapa de erro 2D (heatmap) por imagem
mse_maps = np.mean(np.square(X_test - reconstructed), axis=-1)

# O "score" de anomalia da imagem inteira é o maior erro encontrado nela
# Se houver um pixel muito errado, a imagem é considerada anômala
anomaly_scores = np.max(mse_maps, axis=(1, 2))

# ==============================================================================
# 5. CURVA ROC E AUC A NÍVEL DE IMAGEM
# ==============================================================================

fpr, tpr, thresholds = roc_curve(y_test, anomaly_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Autoencoder ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
plt.title('Curva ROC - Detecção de Anomalias')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n📊 Image-Level AUC (Area Under Curve): {roc_auc:.4f}")

# ==============================================================================
# 6. VISUALIZAÇÃO DOS RESULTADOS (ORIGINAL -> RECONSTRUÍDA -> MAPA DE CALOR)
# ==============================================================================

def visualize_anomalies(original, recon, heatmap, label, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        # Seleciona aleatoriamente dentro das imagens daquela classe
        idx = np.random.choice(np.where(y_test == label)[0])
        
        # Imagem Original
        axes[i, 0].imshow(original[idx])
        axes[i, 0].set_title(f"Original ({'Normal' if label==0 else 'Defeito'})")
        axes[i, 0].axis('off')
        
        # Imagem Reconstruída
        axes[i, 1].imshow(recon[idx])
        axes[i, 1].set_title("Reconstruída pelo Autoencoder")
        axes[i, 1].axis('off')
        
        # Mapa de Calor (Heatmap)
        im = axes[i, 2].imshow(heatmap[idx], cmap='jet', vmin=0, vmax=np.max(heatmap))
        axes[i, 2].set_title(f"Mapa de Erro (Score: {anomaly_scores[idx]:.4f})")
        axes[i, 2].axis('off')
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.show()

print("\nExibindo reconstrução de garrafas NORMAIS...")
visualize_anomalies(X_test, reconstructed, mse_maps, label=0, num_samples=2)

print("\nExibindo reconstrução de garrafas DEFEITUOSAS (Observe o Mapa de Calor)...")
visualize_anomalies(X_test, reconstructed, mse_maps, label=1, num_samples=3)

print("=" * 80)
print("✅ FASE 2 CONCLUÍDA: O Autoencoder consegue localizar as anomalias!")
print("=" * 80)
