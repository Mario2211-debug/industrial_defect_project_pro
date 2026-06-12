"""
================================================================================
PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS - FASE 3: U-NET
================================================================================
Disciplina: Visão por Computador
Dataset: MVTec AD (bottle)
Objetivo: Segmentação a nível de pixel (Onde exatamente está o defeito?)

CONCEITO (U-NET):
Diferente da classificação da Fase 1 (Dizer SE tem defeito) e da reconstrução 
da Fase 2 (Encontrar por erro matemático), a Fase 3 tenta PINTAR o defeito.
Para isso, usamos uma rede "U-Net", que recebe uma imagem e cospe outra imagem
(uma máscara binária), onde Branco = Defeito e Preto = Fundo/Normal.

Nesta fase usamos as imagens da pasta "ground_truth" (o "gabarito" dos defeitos)
como o nosso alvo (target) durante o treinamento.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

# Suprimir warnings
tf.get_logger().setLevel('ERROR')

print("=" * 80)
print("FASE 3: SEGMENTAÇÃO A NÍVEL DE PIXEL COM U-NET")
print("=" * 80)

# ==============================================================================
# 1. CONFIGURAÇÕES E CARREGAMENTO DE DADOS COM MÁSCARAS
# ==============================================================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 20

BASE_PATH = "/home/mafonso/Documents/Mestrado UPT/Visão por computador/UPT_VPC/industrial_defect_project_pro/dataset/bottle"
TEST_PATH = os.path.join(BASE_PATH, "test")
GT_PATH = os.path.join(BASE_PATH, "ground_truth")

# Para segmentação supervisionada, precisamos de imagens COM defeito e suas MÁSCARAS.
# No MVTec AD, o ground_truth só existe para as imagens de teste defeituosas.
# Vamos pegar as imagens de "broken_large", "broken_small" e "contamination".

def load_data_and_masks(category, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    masks = []
    
    img_folder = os.path.join(TEST_PATH, category)
    mask_folder = os.path.join(GT_PATH, category)
    
    if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
        return np.array([]), np.array([])
    
    # As máscaras no MVTec AD geralmente têm o sufixo _mask
    for filename in sorted(os.listdir(img_folder)):
        if filename.endswith(('.png', '.jpg')):
            # Carrega a imagem original
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path).resize(img_size)
            images.append(np.array(img) / 255.0)
            
            # Carrega a máscara correspondente
            base_name = os.path.splitext(filename)[0]
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(mask_folder, mask_filename)
            
            if os.path.exists(mask_path):
                # Máscaras são em escala de cinza (L)
                mask = Image.open(mask_path).convert('L').resize(img_size)
                # Binariza a máscara: pixel > 0 vira 1 (Defeito)
                mask_array = np.array(mask)
                mask_array = (mask_array > 0).astype(np.float32)
                # Adiciona uma dimensão no final para compatibilidade com o Keras (128, 128, 1)
                masks.append(np.expand_dims(mask_array, axis=-1))
            else:
                # Se não tem máscara, assume-se que é tudo preto (normal)
                masks.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)))
                
    return np.array(images), np.array(masks)

print("\nCarregando Imagens e Máscaras (Ground Truth)...")
X_bl, y_bl = load_data_and_masks("broken_large")
X_bs, y_bs = load_data_and_masks("broken_small")
X_ct, y_ct = load_data_and_masks("contamination")

# Juntar tudo (O nosso "Treino" e "Teste" virão daqui apenas para demonstração)
X_all = np.concatenate([X_bl, X_bs, X_ct])
y_all = np.concatenate([y_bl, y_bs, y_ct])

print(f"Total de pares (Imagem / Máscara) carregados: {len(X_all)}")

# Em um cenário real, dividiríamos em treino/validação. 
# Aqui, como o dataset ground_truth é pequeno (apenas os testes), 
# vamos treinar e visualizar com o mesmo conjunto apenas para fins educacionais da U-Net.

# ==============================================================================
# 2. ARQUITETURA DA U-NET
# ==============================================================================

print("\n" + "=" * 80)
print("CONSTRUINDO A U-NET")
print("=" * 80)

# A U-Net tem um caminho de contração (Encoder) e um de expansão (Decoder)
# O segredo são as "Skip Connections" (concatenate), que ligam o Encoder direto no Decoder

inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

# --- ENCODER (Contração) ---
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

# --- BOTTLENECK (O Fundo do "U") ---
c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

# --- DECODER (Expansão) ---
u5 = UpSampling2D((2, 2))(c4)
u5 = concatenate([u5, c3]) # SKIP CONNECTION: Junta a expansão com a característica do Encoder!
c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

u6 = UpSampling2D((2, 2))(c5)
u6 = concatenate([u6, c2]) # SKIP CONNECTION
c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

u7 = UpSampling2D((2, 2))(c6)
u7 = concatenate([u7, c1]) # SKIP CONNECTION
c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

# CAMADA DE SAÍDA
# 1 canal apenas (preto/branco), sigmoid para dar probabilidade de cada pixel ser um defeito
outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

unet = Model(inputs=[inputs], outputs=[outputs])

# Para segmentação, usamos o "binary_crossentropy" focado pixel a pixel
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
unet.summary()

# ==============================================================================
# 3. TREINAMENTO
# ==============================================================================

print("\nTreinando a U-Net para pintar as máscaras de defeito...")
history = unet.fit(
    X_all, y_all, 
    validation_split=0.1, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS,
    verbose=1
)

# ==============================================================================
# 4. VISUALIZAÇÃO DOS RESULTADOS (ORIGINAL -> GABARITO -> PREVISÃO DA REDE)
# ==============================================================================

print("\n" + "=" * 80)
print("VISUALIZANDO A SEGMENTAÇÃO DE DEFEITOS")
print("=" * 80)

# Previsões da rede (Mapas de probabilidade de defeito por pixel)
preds = unet.predict(X_all)

# Binarizar a previsão (Prob > 50% = Defeito)
preds_bin = (preds > 0.5).astype(np.float32)

def plot_segmentation_results(images, true_masks, pred_masks, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # 1. Imagem Original
        axes[i, 0].imshow(images[idx])
        axes[i, 0].set_title("Imagem com Defeito")
        axes[i, 0].axis('off')
        
        # 2. Gabarito (Ground Truth)
        axes[i, 1].imshow(true_masks[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title("Gabarito (Ground Truth)")
        axes[i, 1].axis('off')
        
        # 3. Previsão da U-Net
        axes[i, 2].imshow(pred_masks[idx].squeeze(), cmap='gray')
        axes[i, 2].set_title("Previsão da U-Net")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

plot_segmentation_results(X_all, y_all, preds_bin, num_samples=3)

# ==============================================================================
# 5. CÁLCULO DO DICE SCORE E IoU (Métricas de Segmentação)
# ==============================================================================

# Dice Score = 2 * Área de Interseção / (Área Predita + Área Real)
# IoU (Intersection over Union) = Interseção / União

intersection = np.logical_and(y_all, preds_bin)
union = np.logical_or(y_all, preds_bin)

iou_score = np.sum(intersection) / np.sum(union)
dice_score = 2 * np.sum(intersection) / (np.sum(preds_bin) + np.sum(y_all))

print("\n📊 MÉTRICAS DE SEGMENTAÇÃO (A NÍVEL DE PIXEL):")
print(f"   IoU (Intersection over Union): {iou_score:.4f}")
print(f"   Dice Score (F1-Score para pixels): {dice_score:.4f}")

if dice_score > 0.7:
    print("   ✅ Muito Bom - A rede está pintando o defeito quase exatamente onde ele está.")
elif dice_score > 0.4:
    print("   ⚠️ Razoável - A rede localiza a região do defeito, mas as bordas podem estar imperfeitas.")
else:
    print("   ❌ Ruim - A rede tem dificuldade em localizar os pixels corretos. (Tente mais épocas ou mais dados)")

print("\n" + "=" * 80)
print("✅ FASE 3 CONCLUÍDA: A U-Net segmenta a anomalia a nível de pixel!")
print("=" * 80)
