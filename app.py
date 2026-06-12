"""
================================================================================
PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS - INSPEÇÃO VISUAL AUTOMÁTICA
================================================================================
Disciplina: Visão por Computador
Dataset: MVTec AD (bottle)
Objetivo: Classificar produtos como normais ou defeituosos

================================================================================
INTRODUÇÃO E FUNDAMENTAÇÃO TEÓRICA
================================================================================

1. CONTEXTO DO PROBLEMA:
   Em ambientes industriais, a inspeção visual de qualidade é crucial para garantir
   que produtos não apresentem defeitos como rachaduras, amassados, contaminações
   ou peças faltantes. Tradicionalmente feita por humanos, esse processo pode ser
   automatizado usando técnicas de Visão por Computador e Aprendizado Profundo.

2. TIPO DE APRENDIZADO: SUPERVISIONADO (classificação)
   Conforme visto no material DeepLearning1.pptx, o aprendizado supervisionado
   utiliza dados rotulados para treinar o modelo. Neste caso, temos imagens
   de garrafas classificadas como "normais" ou "defeituosas" (com rachaduras).

   O processo de aprendizado supervisionado segue:
   - Fornecemos ao modelo inputs (imagens) e outputs esperados (rótulos)
   - O modelo compara sua previsão com o valor real
   - O erro é calculado via função de custo (categorical_crossentropy)
   - O ajuste dos pesos é feito via backpropagation e gradient descent

3. ARQUITETURA: REDE NEURAL CONVOLUCIONAL (CNN)
   Conforme explicado no material DeepLearning2.pptx, CNNs são inspiradas no
   córtex visual de mamíferos (estudos de Hubel e Wiesel, Nobel 1981) e foram
   popularizadas por Yann LeCun com a LeNet-5 em 1998 para classificar dígitos
   MNIST.

   Por que CNNs para imagens?
   - Imagens têm alta dimensionalidade (ex: 128x128x3 = 49.152 pixels)
   - Uma rede densa (DNN) teria milhões de parâmetros, tornando o treino inviável
   - CNNs aproveitam que pixels próximos são mais correlacionados entre si
   - Usam operações de convolução com filtros/kernels que detectam características

4. COMPONENTES DA CNN UTILIZADA:
   
   a) CONVOLUÇÃO (Conv2D):
      - Aplica filtros/kernels que deslizam sobre a imagem
      - Cada filtro detecta uma característica específica (bordas, texturas, etc.)
      - No código: filtros 3x3 com ativação ReLU
      - ReLU = max(0, z) - função que acelera o aprendizado (vista nas matérias)

   b) MAX POOLING (MaxPool2D):
      - Reduz a dimensionalidade espacial da imagem
      - Mantém as características mais importantes (valor máximo)
      - Reduz memória e parâmetros, prevenindo overfitting
      - Ex: pool 2x2 com stride 2 reduz 75% dos dados

   c) FLATTEN:
      - "Achata" o tensor 3D em um vetor 1D
      - Prepara os dados para as camadas densas (fully connected)

   d) CAMADAS DENSAS (Dense):
      - Fully connected layers - todos neurônios conectados
      - Camada oculta: 128 neurônios com ativação ReLU
      - Camada de saída: 10 neurônios (não usado aqui) com Softmax

   e) SOFTMAX:
      - Converte os outputs em probabilidades que somam 1
      - Garante uma distribuição bem comportada para classificação

5. MÉTRICAS DE AVALIAÇÃO (vistas em DeepLearning1.pptx):
   - Accuracy: (TP+TN)/Total - taxa de acertos global
   - Loss: categorical_crossentropy - mede o erro da previsão
   - Matriz de confusão: mostra TP, TN, FP, FN
   - Precisão e Recall serão calculados para análise detalhada

6. OTIMIZADOR: ADAM
   - Versão melhorada do Gradient Descent
   - Ajusta a taxa de aprendizado automaticamente
   - Combina vantagens do Momentum e RMSprop

7. PREVENÇÃO DE OVERFITTING:
   - Validation split (10% dos dados de treino)
   - MaxPooling reduz parâmetros
   - Dropout poderia ser adicionado (conforme mencionado nas matérias)
"""

# ==============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# TensorFlow / Keras - Framework para Deep Learning
# Usamos tf.keras (recomendado) em vez de keras diretamente
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 80)
print("PROJETO 2: DETECÇÃO DE DEFEITOS INDUSTRIAIS")
print("Dataset: MVTec AD - Bottle")
print("=" * 80)

# ==============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 80)

"""
PRÉ-PROCESSAMENTO DE IMAGENS:
Conforme visto no material DeepLearning1.pptx, antes de treinar um modelo,
precisamos:
1. Carregar os dados
2. Normalizar os valores dos pixels (0-255 -> 0-1)
3. Separar em treino e teste
4. Aplicar One-Hot Encoding nos rótulos
"""

# Diretórios do dataset MVTec AD
# Estrutura esperada:
# dataset/
#   bottle/
#     train/
#       good/          - imagens normais (treino)
#       broken_small/  - defeito: pequenas quebras
#       broken_large/  - defeito: grandes quebras
#       contamination/ - defeito: contaminação
#     test/
#       good/          - imagens normais (teste)
#       broken_small/  - defeito
#       broken_large/  - defeito
#       contamination/ - defeito

# Configurações
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
NUM_CLASSES = 2  # Normal vs Defeituoso

print("\nCarregando dataset MVTec AD...")
print("Diretório base: dataset/bottle/")

# Criar geradores de dados com aumento de dados (data augmentation)
# Data augmentation: técnicas para aumentar artificialmente o dataset
# (rotações, zoom, etc.) - mencionado em aula como preventivo de overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,      # Normalização: valores de pixel entre 0 e 1
    rotation_range=20,   # Rotação aleatória de até 20 graus
    zoom_range=0.15,     # Zoom aleatório
    width_shift_range=0.2,  # Deslocamento horizontal
    height_shift_range=0.2, # Deslocamento vertical
    validation_split=0.2    # 20% dos dados para validação
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normalização

# Carregar imagens de TREINO
# label_mode='binary' porque temos duas classes: normal (0) e defeito (1)
train_generator = train_datagen.flow_from_directory(
    'dataset/bottle/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',   # Classificação binária: normal vs defeito
    subset='training',     # Parte de treino (80%)
    shuffle=True,
    classes=['good', 'broken_small']  # Simplificando: só um tipo de defeito
)

# Carregar imagens de VALIDAÇÃO
validation_generator = train_datagen.flow_from_directory(
    'dataset/bottle/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',   # Parte de validação (20%)
    shuffle=True,
    classes=['good', 'broken_small']
)

# Carregar imagens de TESTE
test_generator = test_datagen.flow_from_directory(
    'dataset/bottle/test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,  # Importante: não shuffle para avaliação consistente
    classes=['good', 'broken_small']
)

print(f"\nClasses encontradas: {train_generator.class_indices}")
print(f"Total de imagens de treino: {train_generator.samples}")
print(f"Total de imagens de validação: {validation_generator.samples}")
print(f"Total de imagens de teste: {test_generator.samples}")

# ==============================================================================
# 2. VISUALIZAÇÃO EXPLORATÓRIA DOS DADOS
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 2: VISUALIZAÇÃO EXPLORATÓRIA DOS DADOS")
print("=" * 80)

# Função para visualizar imagens do dataset
def visualize_samples(generator, num_samples=5):
    """
    Exibe exemplos de imagens do dataset.
    Importante para entender com que tipo de dado estamos trabalhando.
    """
    images, labels = next(generator)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        axes[i].imshow(images[i])
        title = "NORMAL" if labels[i] == 0 else "DEFEITO"
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.suptitle("Exemplos de Imagens do Dataset MVTec AD", fontsize=14)
    plt.tight_layout()
    plt.show()

print("\nVisualizando amostras do dataset...")
# Recriar generator para não consumir os dados
temp_gen = train_datagen.flow_from_directory(
    'dataset/bottle/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['good', 'broken_small']
)
visualize_samples(temp_gen)

# ==============================================================================
# 3. CONSTRUÇÃO DO MODELO CNN
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 3: CONSTRUÇÃO DA REDE NEURAL CONVOLUCIONAL (CNN)")
print("=" * 80)

"""
ARQUITETURA DA CNN (inspirada na LeNet-5, vista em DeepLearning2.pptx):

Camada 1: Conv2D(32, 3x3, ReLU) + MaxPool2D(2x2)
   - 32 filtros de 3x3 detectam características básicas (bordas, texturas)
   - MaxPool reduz dimensões pela metade

Camada 2: Conv2D(64, 3x3, ReLU) + MaxPool2D(2x2)
   - 64 filtros detectam características mais complexas
   - Nova redução dimensional

Camada 3: Conv2D(128, 3x3, ReLU) + MaxPool2D(2x2)
   - 128 filtros para características de alto nível

Camada 4: Flatten() - Achata o tensor para vetor

Camada 5: Dense(128, ReLU) - Camada totalmente conectada
   - Aprende combinações das características detectadas

Camada 6: Dropout(0.5) - Regularização (opcional, visto em aula)

Camada 7: Dense(1, Sigmoid) - Saída binária (normal=0, defeito=1)

VANTAGENS DESTA ARQUITETURA:
- Número reduzido de parâmetros comparado a uma DNN
- Invariância a translação devido às convoluções
- Hierarquia de características (bordas -> formas -> objetos)
"""

def build_cnn_model():
    """
    Constrói o modelo Sequential com camadas Conv2D, MaxPool2D, Flatten e Dense.
    Sequential: modelo linear onde camadas são empilhadas sequencialmente.
    """
    
    model = Sequential(name="CNN_Classifier_Industrial_Defect")
    
    # INPUT SHAPE: (128, 128, 3) - imagem 128x128 com 3 canais (RGB)
    
    # ===== ENCODER (Extrai características) =====
    # Primeira camada convolucional
    # Conv2D: aplica 32 filtros de tamanho 3x3 com ativação ReLU
    # ReLU = max(0, x) - não-linearidade que permite aprendizado complexo
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    # MaxPooling: reduz resolução pela metade (128x128 -> 64x64)
    # Mantém a característica mais forte em cada região 2x2
    model.add(MaxPool2D((2, 2)))
    
    # Segunda camada convolucional
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))  # 64x64 -> 32x32
    
    # Terceira camada convolucional (mais filtros para características complexas)
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))  # 32x32 -> 16x16
    
    # ===== TRANSITION (Prepara para classificação) =====
    # Flatten: transforma tensor 3D em vetor 1D
    # Exemplo: 16x16x128 = 32.768 valores -> vetor de 32.768
    model.add(Flatten())
    
    # ===== DECODER / CLASSIFICADOR =====
    # Fully Connected Layer com 128 neurônios e ativação ReLU
    # Aprende combinações não-lineares das características
    model.add(Dense(128, activation='relu'))
    
    # Dropout: previne overfitting "desligando" aleatoriamente 50% dos neurônios
    # Durante o treino, diferentes sub-redes são treinadas a cada época
    # Isso força a rede a não depender de neurônios específicos
    model.add(Dropout(0.5))
    
    # Camada de saída: 1 neurônio com ativação Sigmoid
    # Sigmoid: σ(z) = 1/(1+e^(-z)) -> output entre 0 e 1
    # Para classificação binária: valor > 0.5 = DEFEITO, < 0.5 = NORMAL
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Construir o modelo
cnn_model = build_cnn_model()

# Exibir arquitetura do modelo
print("\nArquitetura da CNN:")
print("-" * 50)
cnn_model.summary()

# ==============================================================================
# 4. COMPILAÇÃO DO MODELO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 4: COMPILAÇÃO DO MODELO")
print("=" * 80)

"""
COMPILAÇÃO: Configura o modelo para treinamento
- optimizer='adam': Adam (Adaptive Moment Estimation) é uma versão melhorada
  do Gradient Descent. Ajusta a taxa de aprendizado automaticamente.
  
- loss='binary_crossentropy': Função de custo para classificação binária.
  Mede a diferença entre a previsão (ŷ) e o valor real (y).
  Fórmula: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  Penaliza mais erros grandes, acelerando o aprendizado.
  
- metrics=['accuracy']: Métrica de avaliação principal - taxa de acertos.
"""

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\nModelo compilado com:")
print("  - Optimizer: Adam (Gradient Descent adaptativo)")
print("  - Loss: Binary Cross-Entropy")
print("  - Métricas: Accuracy, Precision, Recall")

# ==============================================================================
# 5. TREINAMENTO DO MODELO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 5: TREINAMENTO DA REDE NEURAL")
print("=" * 80)

"""
TREINAMENTO: Processo de aprendizado onde o modelo ajusta seus pesos

BACKPROPAGATION (visto em DeepLearning1.pptx):
1. Forward pass: dados entram na rede, produzindo uma previsão
2. Cálculo do erro (loss) comparando previsão com valor real
3. Backward pass: erro é propagado de volta através da rede
4. Gradient Descent: pesos são ajustados na direção que minimiza o erro

ÉPOCA: Uma passagem completa por TODO o dataset de treino
BATCH: Subconjunto de dados processado antes de atualizar os pesos
VALIDATION_SPLIT: 20% dos dados de treino são reservados para validação
"""

EPOCHS = 15

print(f"\nIniciando treinamento por {EPOCHS} épocas...")
print("Processo de aprendizado:")
print("  1. Forward pass: calcula previsão para cada imagem")
print("  2. Calcula erro (loss) comparando com rótulo real")
print("  3. Backward pass: propaga erro de volta pela rede")
print("  4. Atualiza pesos via Gradient Descent (optimizer Adam)")
print("  5. Repete para cada batch até completar uma época")
print("  6. Ao final da época, avalia performance na validação")
print("-" * 60)

# Treinar o modelo
# verbose=1 mostra barra de progresso durante o treino
history = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# ==============================================================================
# 6. VISUALIZAÇÃO DO TREINAMENTO (Learning Curves)
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 6: ANÁLISE DO TREINAMENTO - CURVAS DE APRENDIZADO")
print("=" * 80)

"""
CURVAS DE APRENDIZADO: Mostram evolução do erro (loss) e acurácia ao longo das épocas

Interpretação:
- Loss decrescente: modelo está aprendendo
- Loss estabiliza: convergiu para mínimo da função de custo
- Overfitting: loss treino cai, loss validação sobe
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Loss (Erro)
axes[0].plot(history.history['loss'], label='Treino', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validação', linewidth=2)
axes[0].set_title('Curva de Loss (Erro)', fontsize=14)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Binary Cross-Entropy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico de Acurácia
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
# 7. AVALIAÇÃO DO MODELO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 7: AVALIAÇÃO DO MODELO COM DADOS DE TESTE")
print("=" * 80)

"""
AVALIAÇÃO: Testa o modelo em dados que ele NUNCA viu
Métricas calculadas:
- Loss: erro médio nas previsões
- Accuracy: percentual de acertos (TP+TN)/Total
- Precision: TP/(TP+FP) - qualidade das previsões positivas
- Recall: TP/(TP+FN) - capacidade de encontrar todos os defeitos
"""

print("\nAvaliando modelo no conjunto de teste...")
print("(Imagens que o modelo nunca viu durante o treinamento)")
print("-" * 50)

test_loss, test_accuracy, test_precision, test_recall = cnn_model.evaluate(test_generator)

print(f"\n📊 RESULTADOS NO CONJUNTO DE TESTE:")
print(f"   Loss (Erro):     {test_loss:.4f}")
print(f"   Accuracy:        {test_accuracy:.4f} = {test_accuracy*100:.2f}%")
print(f"   Precision:       {test_precision:.4f}")
print(f"   Recall (Sensibilidade): {test_recall:.4f}")

# Cálculo do F1-Score (média harmônica entre precisão e recall)
# Conforme visto em DeepLearning1.pptx, F1 combina precision e recall
if test_precision + test_recall > 0:
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"   F1-Score:        {f1_score:.4f}")

print(f"\nAnálise da Acurácia: {test_accuracy*100:.2f}% das imagens foram classificadas corretamente")
print("Isso significa que o modelo identifica corretamente se uma garrafa tem defeito ou não.")

# ==============================================================================
# 8. MATRIZ DE CONFUSÃO
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 8: MATRIZ DE CONFUSÃO")
print("=" * 80)

"""
MATRIZ DE CONFUSÃO (vista em DeepLearning1.pptx):

Estrutura:
              Predito Negativo  Predito Positivo
Real Negativo       TN                FP
Real Positivo       FN                TP

- TP (True Positive):  modelo acertou DEFEITO
- TN (True Negative):  modelo acertou NORMAL
- FP (False Positive): modelo errou (disse DEFEITO, era NORMAL)
- FN (False Negative): modelo errou (disse NORMAL, era DEFEITO)

Interpretação:
- FP: alarme falso - custo operacional desnecessário
- FN: defeito não detectado - PODE SER GRAVE dependendo do produto
"""

print("\nGerando matriz de confusão...")

# Obter previsões para o conjunto de teste
test_generator.reset()  # Resetar para começar do início
Y_pred = cnn_model.predict(test_generator)
y_pred_binary = (Y_pred > 0.5).astype(int).flatten()

# Obter rótulos reais
y_true = test_generator.classes[:len(y_pred_binary)]

# Calcular matriz de confusão
cm = confusion_matrix(y_true, y_pred_binary)

# Visualizar matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'DEFEITO'],
            yticklabels=['NORMAL', 'DEFEITO'])
plt.title('Matriz de Confusão - Classificação de Defeitos', fontsize=14)
plt.xlabel('Predito', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.tight_layout()
plt.show()

# Extrair valores da matriz de confusão
TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print("\n📋 MATRIZ DE CONFUSÃO:")
print(f"   True Negatives (NORMAL → NORMAL):     {TN}")
print(f"   False Positives (NORMAL → DEFEITO):   {FP}  (Alarme falso)")
print(f"   False Negatives (DEFEITO → NORMAL):   {FN}  (Defeito não detectado!)")
print(f"   True Positives (DEFEITO → DEFEITO):   {TP}")

# ==============================================================================
# 9. CURVA ROC E AUC
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 9: CURVA ROC E AUC")
print("=" * 80)

"""
CURVA ROC (Receiver Operating Characteristic):
- Mostra o trade-off entre True Positive Rate (Recall) e False Positive Rate
- TPR = TP/(TP+FN) - capacidade de detectar defeitos
- FPR = FP/(FP+TN) - taxa de alarmes falsos

AUC (Area Under Curve):
- Mede a capacidade geral do modelo de distinguir classes
- AUC = 1.0: classificador perfeito
- AUC = 0.5: classificador aleatório (chute)
- Quanto maior, melhor
"""

# Resetar generator para obter todas as previsões
test_generator.reset()
y_pred_proba = cnn_model.predict(test_generator).flatten()
y_true_all = test_generator.classes[:len(y_pred_proba)]

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_true_all, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotar curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CNN (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR) - Taxa de Alarmes Falsos', fontsize=12)
plt.ylabel('True Positive Rate (TPR) - Recall', fontsize=12)
plt.title('Curva ROC - Avaliação do Classificador de Defeitos', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n📊 AUC (Area Under Curve): {roc_auc:.4f}")
print("Interpretação da AUC:")
if roc_auc >= 0.9:
    print("   ✅ Excelente - Modelo tem alta capacidade discriminatória")
elif roc_auc >= 0.8:
    print("   👍 Bom - Modelo funciona bem para maioria dos casos")
elif roc_auc >= 0.7:
    print("   ⚠️ Razoável - Pode precisar de melhorias")
else:
    print("   ❌ Ruim - Modelo não está aprendendo adequadamente")

# ==============================================================================
# 10. VISUALIZAÇÃO DE RESULTADOS (Previsões)
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 10: VISUALIZAÇÃO DAS PREVISÕES")
print("=" * 80)

"""
Esta seção mostra exemplos reais de como o modelo classifica as imagens,
incluindo:
- ACERTOS: quando previsão = valor real
- ERROS: quando o modelo se confunde (FP ou FN)

Importante para entender onde o modelo está acertando e onde está falhando.
"""

def visualize_predictions(generator, model, num_samples=8):
    """
    Visualiza as previsões do modelo em imagens de teste.
    """
    generator.reset()
    images, labels = next(generator)
    predictions = model.predict(images, verbose=0)
    pred_labels = (predictions > 0.5).astype(int).flatten()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        
        # Definir cores e textos
        true_label = "NORMAL" if labels[i] == 0 else "DEFEITO"
        pred_label = "NORMAL" if pred_labels[i] == 0 else "DEFEITO"
        confidence = predictions[i][0] if pred_labels[i] == 1 else 1 - predictions[i][0]
        
        # Cor do título: verde se acertou, vermelho se errou
        is_correct = (labels[i] == pred_labels[i])
        color = 'green' if is_correct else 'red'
        
        title = f"Real: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle("Previsões do Modelo em Imagens de Teste", fontsize=16)
    plt.tight_layout()
    plt.show()

print("\nVisualizando previsões do modelo em imagens de teste...")
visualize_predictions(test_generator, cnn_model)

# ==============================================================================
# 11. RELATÓRIO FINAL DE CLASSIFICAÇÃO
# ==============================================================================

print("\n" + "=" * 80)
print("RELATÓRIO FINAL - CLASSIFICAÇÃO DE DEFEITOS INDUSTRIAIS")
print("=" * 80)

print("""
RESUMO DOS RESULTADOS:
""")

print(f"📈 MÉTRICAS PRINCIPAIS:")
print(f"   • Acurácia:  {test_accuracy*100:.2f}%")
print(f"   • Precisão:  {test_precision:.4f}")
print(f"   • Recall:    {test_recall:.4f}")
print(f"   • AUC:       {roc_auc:.4f}")

# Análise dos resultados
print(f"""
🔍 ANÁLISE DOS RESULTADOS:

1. ACURÁCIA = {test_accuracy*100:.2f}%
   O modelo classificou corretamente {test_accuracy*100:.2f}% das imagens.
   """)

if test_accuracy > 0.9:
    print("   ✅ Excelente performance! O modelo é muito confiável.")
elif test_accuracy > 0.8:
    print("   👍 Boa performance. O modelo funciona bem na maioria dos casos.")
else:
    print("   ⚠️ Performance moderada. Pode haver espaço para melhorias.")

# Análise de false positives e false negatives
fp_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
fn_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"""
2. ANÁLISE DE ERROS:
   • False Positives (Alarmes Falsos): {FP} imagens boas classificadas como defeito
     Taxa: {fp_rate:.2%} das imagens normais
     
   • False Negatives (Defeitos não detectados): {FN} imagens defeituosas classificadas como boas
     Taxa: {fn_rate:.2%} das imagens defeituosas
""")

if FN > FP:
    print("   ⚠️ ATENÇÃO: Mais defeitos não detectados do que alarmes falsos.")
    print("   → Em inspeção industrial, NÃO DETECTAR um defeito pode ser mais grave")
    print("   → que gerar um alarme falso. Considere ajustar o threshold.")
else:
    print("   ✅ O modelo tem mais alarmes falsos do que defeitos não detectados.")
    print("   → Isso é desejável em inspeção industrial (prefere-se falso alarme a")
    print("   → deixar passar um produto defeituoso).")

print(f"""
3. APLICAÇÃO PRÁTICA:
   Este modelo pode ser usado em uma linha de produção automatizada para:
   - Inspecionar garrafas em tempo real
   - Sinalizar produtos defeituosos para remoção
   - Reduzir a necessidade de inspeção visual humana
""")

# ==============================================================================
# 12. SALVAR O MODELO (Opcional)
# ==============================================================================

print("\n" + "=" * 80)
print("FASE 11: SALVANDO O MODELO TREINADO")
print("=" * 80)

# Descomente para salvar o modelo
# cnn_model.save('industrial_defect_detector.h5')
# print("✅ Modelo salvo como 'industrial_defect_detector.h5'")

print("\n" + "=" * 80)
print("✅ TRABALHO CONCLUÍDO COM SUCESSO!")
print("=" * 80)

"""
CONCLUSÃO:

Este projeto implementou um sistema de detecção de defeitos industriais usando
uma Rede Neural Convolucional (CNN), conforme ensinado nas matérias:

1. Aprendizado Supervisionado (DeepLearning1.pptx):
   - Usamos dados rotulados (normal vs defeito)
   - Backpropagation para ajustar pesos
   - Gradient Descent (Adam) para otimização

2. Arquitetura CNN (DeepLearning2.pptx):
   - Camadas Conv2D para extração de características
   - MaxPooling para redução dimensional
   - Flatten + Dense para classificação
   - Dropout para prevenir overfitting

3. Métricas de Avaliação (DeepLearning1.pptx):
   - Matriz de Confusão, Precisão, Recall, F1-Score
   - Curva ROC e AUC para análise discriminatória

PRÓXIMOS PASSOS PARA O PROJETO COMPLETO:
- Implementar Autoencoder para detecção de anomalias (apenas imagens normais)
- U-Net para segmentação pixel-level
- Extração de features com ResNet pré-treinada
- Comparar abordagens supervisionadas vs não-supervisionadas
"""