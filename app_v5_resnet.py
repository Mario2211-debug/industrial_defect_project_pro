"""
================================================================================
PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS - FASE 4: RESNET + TRANSFER LEARNING
================================================================================
Disciplina: Visão por Computador
Dataset: MVTec AD (bottle)
Objetivo: Extração de Features (Características) com Rede Pré-Treinada

CONCEITO (TRANSFER LEARNING & FEATURE EXTRACTION):
Em vez de treinarmos uma rede do zero (como no Autoencoder ou na CNN v2), 
nós vamos "pegar emprestado" o cérebro da ResNet50, uma rede monstruosa da 
Microsoft treinada em mais de 1 milhão de imagens do mundo real (ImageNet).

Como a ResNet já é mestre em identificar texturas, cores e formas, nós 
vamos passar as nossas imagens por ela e extrair o "vetor de características" 
(embeddings). Depois, usamos um algoritmo clássico de Machine Learning 
(One-Class SVM ou K-Nearest Neighbors) apenas para dizer: 
"Essa característica parece com as das garrafas normais do treino?"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from sklearn.svm import OneClassSVM

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Suprimir warnings
tf.get_logger().setLevel('ERROR')

print("=" * 80)
print("FASE 4: DETECÇÃO DE ANOMALIAS COM RESNET-50 (TRANSFER LEARNING)")
print("=" * 80)

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS (Reaproveitando a lógica de Anomalia)
# ==============================================================================
# O MVTec AD só tem imagens normais no Treino!

IMG_HEIGHT = 224 # A ResNet50 prefere imagens 224x224
IMG_WIDTH = 224

BASE_PATH = "/home/mafonso/Documents/Mestrado UPT/Visão por computador/UPT_VPC/industrial_defect_project_pro/dataset/bottle"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

def load_images_for_resnet(folder_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    if not os.path.exists(folder_path): return np.array([])
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            img_array = np.array(img)
            # Ao invés de dividir por 255, usamos o preprocessamento específico da ResNet
            img_preprocessed = preprocess_input(img_array)
            images.append(img_preprocessed)
    return np.array(images)

print("\nCarregando imagens de TREINO (Apenas NORMAS)...")
X_train = load_images_for_resnet(os.path.join(TRAIN_PATH, "good"))

print("Carregando imagens de TESTE...")
X_test_good = load_images_for_resnet(os.path.join(TEST_PATH, "good"))
X_test_broken_large = load_images_for_resnet(os.path.join(TEST_PATH, "broken_large"))
X_test_broken_small = load_images_for_resnet(os.path.join(TEST_PATH, "broken_small"))
X_test_contamination = load_images_for_resnet(os.path.join(TEST_PATH, "contamination"))

# Unir testes
X_test = np.concatenate([X_test_good, X_test_broken_large, X_test_broken_small, X_test_contamination])
# Labels: 1 para Normal (O OneClassSVM usa 1 para normal e -1 para anomalia)
y_test_svm = np.concatenate([np.ones(len(X_test_good)), 
                             -np.ones(len(X_test_broken_large) + len(X_test_broken_small) + len(X_test_contamination))])
# Labels para ROC: 0 Normal, 1 Anomalia
y_test_roc = np.concatenate([np.zeros(len(X_test_good)), 
                             np.ones(len(X_test_broken_large) + len(X_test_broken_small) + len(X_test_contamination))])

print(f"✅ Treino: {len(X_train)} normais | Teste: {len(X_test)} total ({len(X_test_good)} normais, {len(X_test)-len(X_test_good)} defeitos)")

# ==============================================================================
# 2. CARREGAR A RESNET50 COMO "EXTRATORA DE CARACTERÍSTICAS"
# ==============================================================================

print("\n" + "=" * 80)
print("BAIXANDO/CARREGANDO A RESNET-50 (PODE LEVAR ALGUNS SEGUNDOS)")
print("=" * 80)

# include_top=False: Removemos a camada final (que servia para classificar cães, gatos, carros...)
# weights='imagenet': Carregamos a "inteligência" (pesos) que ela aprendeu na Microsoft/Google
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# GlobalAveragePooling2D: Transforma as matrizes (ex: 7x7x2048) num vetor 1D de 2048 números
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# ==============================================================================
# 3. EXTRAÇÃO DAS CARACTERÍSTICAS (O "DNA" DAS IMAGENS)
# ==============================================================================

print("\nExtraindo o 'DNA' visual das imagens de treino...")
# Isso gera uma matriz de formato (209, 2048) - 209 imagens, cada uma descrita por 2048 números!
features_train = feature_extractor.predict(X_train, batch_size=32)

print("Extraindo o 'DNA' visual das imagens de teste...")
features_test = feature_extractor.predict(X_test, batch_size=32)

# ==============================================================================
# 4. ALGORITMO NÃO-SUPERVISIONADO: ONE-CLASS SVM
# ==============================================================================

print("\n" + "=" * 80)
print("TREINANDO O ONE-CLASS SVM (APRENDENDO A BOLHA DA NORMALIDADE)")
print("=" * 80)

# O One-Class SVM vai olhar para os 2048 números de cada imagem normal
# e tentar desenhar uma "bolha matemática" ao redor deles.
# Qualquer imagem cujo DNA caia fora dessa bolha será considerada DEFEITUOSA!

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05) # nu = taxa esperada de falsos positivos no treino
svm.fit(features_train)

# ==============================================================================
# 5. TESTE E AVALIAÇÃO
# ==============================================================================

# Prever no teste (Retorna 1 para Normal, -1 para Anomalia)
preds_svm = svm.predict(features_test)

# O SVM também dá um "score" numérico de quão longe a imagem está do centro da bolha
# Valores mais negativos = mais anômalo
scores_svm = svm.score_samples(features_test)

# Inverter os scores para a Curva ROC (queremos que valores maiores = anomalia)
anomaly_scores_roc = -scores_svm

print("\n" + "=" * 80)
print("RESULTADOS DO TRANSFER LEARNING (RESNET50 + SVM)")
print("=" * 80)

# Matriz de Confusão
# Vamos converter os labels do SVM para (0=Normal, 1=Defeito)
preds_binary = np.where(preds_svm == 1, 0, 1)

cm = confusion_matrix(y_test_roc, preds_binary)
TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print("\n📋 MATRIZ DE CONFUSÃO (SVM):")
print(f"   True Negatives  (NORMAL → NORMAL):     {TN}")
print(f"   False Positives (NORMAL → DEFEITO):    {FP}  ← Alarmes falsos")
print(f"   False Negatives (DEFEITO → NORMAL):    {FN}  ← Defeitos não detectados")
print(f"   True Positives  (DEFEITO → DEFEITO):   {TP}")

# ==============================================================================
# 6. CURVA ROC
# ==============================================================================

fpr, tpr, thresholds = roc_curve(y_test_roc, anomaly_scores_roc)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ResNet50 ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
plt.title('Curva ROC - ResNet50 (Transfer Learning)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n📊 AUC (Area Under Curve): {roc_auc:.4f}")

if roc_auc > 0.90:
    print("\n✅ INCRÍVEL! Veja como o Transfer Learning superou facilmente os modelos treinados do zero!")
    print("A inteligência prévia da ResNet torna a identificação de defeitos muito mais robusta.")

print("\n" + "=" * 80)
print("✅ FASE 4 CONCLUÍDA: A magia do Transfer Learning e Feature Extraction!")
print("=" * 80)
