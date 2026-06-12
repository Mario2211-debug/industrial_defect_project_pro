"""
================================================================================
PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS - INSPEÇÃO VISUAL AUTOMÁTICA
================================================================================
Disciplina: Visão por Computador
Dataset: MVTec AD (bottle)
Objetivo: Classificar produtos como normais ou defeituosos

ESTRUTURA DO DATASET:
- train/good/          -> Apenas imagens NORMAS para treino (209 imagens)
- test/good/           -> Imagens NORMAS para teste (20 imagens)
- test/broken_large/   -> Imagens com defeito (20 imagens)
- test/broken_small/   -> Imagens com defeito (22 imagens)
- test/contamination/  -> Imagens com defeito (21 imagens)
- ground_truth/        -> Máscaras para localização de defeitos

NOTA: O modelo é treinado APENAS com imagens normais (aprendizado não-supervisionado
para detecção de anomalias). Nesta primeira fase, usaremos aprendizado supervisionado
para classificação binária.
"""

# ==============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suprimir warnings
tf.get_logger().setLevel('ERROR')

print("=" * 80)
print("PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS")
print("Dataset: MVTec AD - Bottle")
print("=" * 80)

# ==============================================================================
# 1. CONFIGURAÇÕES INICIAIS
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 1: CONFIGURAÇÕES INICIAIS")
print("=" * 80)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15

# Caminhos do dataset
BASE_PATH = "/home/mafonso/Documents/Mestrado UPT/Visão por computador/UPT_VPC/industrial_defect_project_pro/dataset/bottle"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
GROUND_TRUTH_PATH = os.path.join(BASE_PATH, "ground_truth")

print(f"Configurações:")
print(f"  - Tamanho das imagens: {IMG_HEIGHT}x{IMG_WIDTH} pixels")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Épocas: {EPOCHS}")
print(f"\nCaminhos:")
print(f"  - Treino: {TRAIN_PATH}")
print(f"  - Teste: {TEST_PATH}")

# Verificar se o dataset existe
if not os.path.exists(BASE_PATH):
    print(f"\n❌ ERRO: Pasta '{BASE_PATH}' não encontrada!")
    print("Certifique-se de que está no diretório correto.")
    exit(1)

# ==============================================================================
# 2. CARREGAMENTO DOS DADOS
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 2: CARREGAMENTO DOS DADOS")
print("=" * 80)

"""
CARREGAMENTO MANUAL DOS DADOS:
Como o train contém apenas imagens normais e o test contém várias categorias,
vamos carregar manualmente para garantir a correta classificação.
"""

def load_images_from_folder(folder_path, label, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Carrega todas as imagens de uma pasta e atribui um rótulo.
    """
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"  ⚠️ Pasta não encontrada: {folder_path}")
        return np.array([]), np.array([])
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                img = img.resize(img_size)
                img = np.array(img) / 255.0  # Normalização
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"  Erro ao carregar {filename}: {e}")
    
    return np.array(images), np.array(labels)

print("\nCarregando imagens de TREINO (apenas imagens normais)...")
train_images, train_labels = load_images_from_folder(
    os.path.join(TRAIN_PATH, "good"), 
    label=0  # 0 = NORMAL
)

print(f"  ✅ Treino: {len(train_images)} imagens NORMAS carregadas")

print("\nCarregando imagens de TESTE...")

# Teste - Imagens Normais
test_normal_images, test_normal_labels = load_images_from_folder(
    os.path.join(TEST_PATH, "good"),
    label=0  # NORMAL
)

# Teste - Imagens com Defeito (todas as categorias)
defect_categories = ['broken_large', 'broken_small', 'contamination']
test_defect_images = []
test_defect_labels = []

for category in defect_categories:
    category_path = os.path.join(TEST_PATH, category)
    if os.path.exists(category_path):
        images, labels = load_images_from_folder(category_path, label=1)  # 1 = DEFEITO
        if len(images) > 0:
            test_defect_images.extend(images)
            test_defect_labels.extend(labels)
            print(f"  ✅ {category}: {len(images)} imagens com DEFEITO")

# Concatenar todas as imagens de teste
test_images = np.concatenate([test_normal_images, np.array(test_defect_images)])
test_labels = np.concatenate([test_normal_labels, np.array(test_defect_labels)])

print(f"\n📊 RESUMO DO DATASET:")
print(f"  Treino: {len(train_images)} imagens (Todas NORMAS)")
print(f"  Teste: {len(test_images)} imagens")
print(f"    - Normais: {len(test_normal_images)}")
print(f"    - Defeitos: {len(test_defect_images)}")

# ==============================================================================
# 3. DIVISÃO TREINO/VALIDAÇÃO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 3: DIVISÃO TREINO/VALIDAÇÃO")
print("=" * 80)

from sklearn.model_selection import train_test_split

# Dividir os dados de treino em treino e validação (80% / 20%)
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels
)

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(X_train)} imagens")
print(f"  Validação: {len(X_val)} imagens")
print(f"  Teste: {len(test_images)} imagens")

# ==============================================================================
# 4. VISUALIZAÇÃO EXPLORATÓRIA DOS DADOS
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 4: VISUALIZAÇÃO EXPLORATÓRIA DOS DADOS")
print("=" * 80)

def visualize_samples(images, labels, num_samples=5):
    """
    Exibe exemplos de imagens do dataset.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        title = "NORMAL" if labels[idx] == 0 else "DEFEITO"
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle("Exemplos de Imagens do Dataset MVTec AD", fontsize=14)
    plt.tight_layout()
    plt.show()

print("\nVisualizando amostras do dataset de TREINO (apenas normais)...")
visualize_samples(X_train, y_train)

print("\nVisualizando amostras do dataset de TESTE...")
visualize_samples(test_images, test_labels)

# ==============================================================================
# 5. CONSTRUÇÃO DO MODELO CNN
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 5: CONSTRUÇÃO DA REDE NEURAL CONVOLUCIONAL (CNN)")
print("=" * 80)

"""
ARQUITETURA DA CNN (inspirada na LeNet-5 apresentada nas matérias):

Justificativa das camadas:
1. Conv2D(32, 3x3): Detecção de características básicas (bordas, texturas)
2. MaxPool2D: Redução dimensional, mantém características importantes
3. Conv2D(64, 3x3): Detecção de características mais complexas
4. Conv2D(128, 3x3): Características de alto nível
5. Flatten: Prepara para classificação
6. Dense(128): Camada totalmente conectada para combinar características
7. Dropout(0.5): Previne overfitting (mencionado nas matérias)
8. Dense(1, sigmoid): Classificação binária (normal vs defeito)

Vantagens sobre DNN:
- Redução drástica de parâmetros
- Invariância a translação
- Hierarquia de características
"""

model = Sequential(name="CNN_Defect_Detector")

# Input shape: (128, 128, 3) - imagem RGB
model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

# Bloco 1: Características básicas
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))  # 128x128 -> 64x64

# Bloco 2: Características intermediárias
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))  # 64x64 -> 32x32

# Bloco 3: Características avançadas
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))  # 32x32 -> 16x16

# Classificador
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularização
model.add(Dense(1, activation='sigmoid'))  # Saída binária

print("\nArquitetura do Modelo CNN:")
print("-" * 50)
model.summary()

# Calcular número total de parâmetros
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"\nTotal de parâmetros: {total_params:,}")
print(f"Parâmetros treináveis: {trainable_params:,}")

# ==============================================================================
# 6. COMPILAÇÃO DO MODELO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 6: COMPILAÇÃO DO MODELO")
print("=" * 80)

"""
COMPILAÇÃO - Configura o modelo para treinamento:

- Optimizer 'adam': Adaptive Moment Estimation
  * Versão melhorada do Gradient Descent
  * Ajusta taxa de aprendizado automaticamente
  * Combina vantagens do Momentum e RMSprop

- Loss 'binary_crossentropy': Função de custo para classificação binária
  * Mede diferença entre previsão e valor real
  * Penaliza mais erros grandes

- Metrics ['accuracy']: Percentual de acertos
"""

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("✅ Modelo compilado com sucesso!")
print("  - Optimizer: Adam (Gradient Descent adaptativo)")
print("  - Loss: Binary Cross-Entropy")
print("  - Métricas: Accuracy, Precision, Recall")

# ==============================================================================
# 7. TREINAMENTO DO MODELO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 7: TREINAMENTO DO MODELO")
print("=" * 80)

"""
PROCESSO DE TREINAMENTO (Backpropagation + Gradient Descent):

Para cada época:
1. Forward pass: Imagem -> rede -> previsão
2. Calcula erro (loss) comparando previsão com label real
3. Backward pass: Propaga o erro de volta pela rede
4. Gradient Descent: Ajusta pesos para minimizar erro
5. Repete para cada batch
"""

print(f"\nIniciando treinamento por {EPOCHS} épocas...")
print(f"Total de batches por época: {len(X_train) // BATCH_SIZE}")
print("-" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# ==============================================================================
# 8. CURVAS DE APRENDIZADO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 8: ANÁLISE DAS CURVAS DE APRENDIZADO")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Treino', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validação', linewidth=2)
axes[0].set_title('Curva de Loss (Erro)', fontsize=14)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Binary Cross-Entropy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Treino', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
axes[1].set_title('Curva de Acurácia', fontsize=14)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Evolução do Aprendizado Durante o Treinamento', fontsize=16)
plt.tight_layout()
plt.show()

# ==============================================================================
# 9. AVALIAÇÃO DO MODELO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 9: AVALIAÇÃO DO MODELO")
print("=" * 80)

print("\nAvaliando modelo no conjunto de teste (dados nunca vistos)...")
print("-" * 50)

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_images, test_labels, verbose=0)

print(f"\n📊 RESULTADOS NO CONJUNTO DE TESTE:")
print(f"   Loss:           {test_loss:.4f}")
print(f"   Accuracy:       {test_accuracy:.4f} = {test_accuracy*100:.2f}%")
print(f"   Precision:      {test_precision:.4f}")
print(f"   Recall:         {test_recall:.4f}")

# F1-Score
f1 = 0
if test_precision + test_recall > 0:
    f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"   F1-Score:       {f1:.4f}")

# ==============================================================================
# 10. MATRIZ DE CONFUSÃO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 10: MATRIZ DE CONFUSÃO")
print("=" * 80)

# Previsões
y_pred_proba = model.predict(test_images, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Matriz de confusão
cm = confusion_matrix(test_labels, y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'DEFEITO'],
            yticklabels=['NORMAL', 'DEFEITO'])
plt.title('Matriz de Confusão - Classificação de Defeitos', fontsize=14)
plt.xlabel('Predito', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.tight_layout()
plt.show()

# Extrair valores
TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print("\n📋 MATRIZ DE CONFUSÃO DETALHADA:")
print(f"   True Negatives  (NORMAL → NORMAL):     {TN}")
print(f"   False Positives (NORMAL → DEFEITO):    {FP}  ← Alarmes falsos")
print(f"   False Negatives (DEFEITO → NORMAL):    {FN}  ← Defeitos não detectados")
print(f"   True Positives  (DEFEITO → DEFEITO):   {TP}")

# ==============================================================================
# 11. CURVA ROC E AUC
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 11: CURVA ROC E AUC")
print("=" * 80)

fpr, tpr, thresholds = roc_curve(test_labels, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CNN (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR) - Taxa de Alarmes Falsos', fontsize=12)
plt.ylabel('True Positive Rate (TPR) - Recall/Sensibilidade', fontsize=12)
plt.title('Curva ROC - Avaliação do Classificador de Defeitos', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n📊 AUC (Area Under Curve): {roc_auc:.4f}")

if roc_auc >= 0.95:
    print("   Classificação: EXCELENTE")
elif roc_auc >= 0.90:
    print("   Classificação: MUITO BOM")
elif roc_auc >= 0.80:
    print("   Classificação: BOM")
elif roc_auc >= 0.70:
    print("   Classificação: RAZOÁVEL")
else:
    print("   Classificação: INSUFICIENTE")

# ==============================================================================
# 12. RELATÓRIO FINAL
# ==============================================================================

print("\n" + "=" * 80)
print("RELATÓRIO FINAL - CLASSIFICAÇÃO DE DEFEITOS INDUSTRIAIS")
print("=" * 80)

print(f"""
RESUMO DOS RESULTADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MÉTRICAS PRINCIPAIS:
   • Acurácia:  {test_accuracy*100:.2f}%
   • Precisão:  {test_precision:.4f}
   • Recall:    {test_recall:.4f}
   • F1-Score:  {f1:.4f}
   • AUC:       {roc_auc:.4f}

📋 ANÁLISE DE ERROS:
   • Alarmes falsos (FP):         {FP} imagens boas classificadas como defeito
   • Defeitos não detectados (FN): {FN} imagens defeituosas classificadas como boas

{'✅ ' if FN <= FP else '⚠️ '}AVALIAÇÃO PARA INSPEÇÃO INDUSTRIAL:
   {'Modelo tem mais alarmes falsos que defeitos não detectados.' if FN <= FP else 'ATENÇÃO: Muitos defeitos não estão sendo detectados!'}
   {'Isso é aceitável pois é melhor inspecionar um falso alarme do que enviar um produto defeituoso.' if FN <= FP else 'Considere ajustar o threshold ou melhorar o modelo.'}

🎯 APLICAÇÃO PRÁTICA:
   Este modelo pode ser integrado a um sistema de visão computacional
   em linha de produção para:
   - Inspecionar garrafas automaticamente
   - Sinalizar produtos defeituosos
   - Reduzir custos com inspeção manual
   - Aumentar consistência da qualidade

📚 CONCEITOS APLICADOS (das matérias):
   1. Aprendizado Supervisionado (DeepLearning1.pptx)
   2. Redes Neurais Convolucionais (DeepLearning2.pptx)
   3. Backpropagation e Gradient Descent
   4. Funções de ativação (ReLU, Sigmoid)
   5. Pooling para redução dimensional
   6. Dropout para regularização
   7. Matriz de confusão e métricas de avaliação
   8. Curva ROC e AUC

🔜 PRÓXIMOS PASSOS (expansão do projeto):
   • Autoencoder para detecção não-supervisionada (treinado apenas com normais)
   • U-Net para segmentação e localização de defeitos
   • Feature extraction com ResNet pré-treinada
   • Comparação supervisionado vs não-supervisionado
   • Pixel-level AUROC e IoU/Dice score
""")

print("=" * 80)
print("✅ PROJETO CONCLUÍDO COM SUCESSO!")
print("=" * 80)

# Opcional: salvar o modelo treinado
# model.save('cnn_defect_detector.h5')
# print("\n💾 Modelo salvo como 'cnn_defect_detector.h5'")