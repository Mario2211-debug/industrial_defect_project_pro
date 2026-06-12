# Tutorial Parte 1: Fundamentos e Classificação Supervisionada com CNN

Bem-vindo ao **Projeto 2: Detecção de Defeitos Industriais**. Este documento serve como um guia abrangente e linha a linha do script `app_v2.py`, explicando não só "o que" o código faz, mas principalmente "por que" ele foi escrito desta forma.

A ideia deste projeto é simular um problema real de inspeção visual numa linha de produção industrial: como ensinar um computador a identificar automaticamente se uma garrafa fabricada está em perfeitas condições (NORMAL) ou se possui algum defeito (DEFEITUOSA).

---

## 1. Fundamentos e Conceitos Básicos

### O Desafio Proposto
Na indústria, a inspeção manual é lenta, cara e suscetível à fadiga humana. Um inspetor humano pode deixar passar uma rachadura quase invisível no final do seu turno. A Visão por Computador resolve esse problema criando sistemas automatizados que analisam imagens em frações de segundo com precisão consistente. 

O desafio aqui é pegar um dataset (conjunto de dados de imagens) chamado **MVTec AD** (focado na categoria "bottle" - garrafas) e treinar um modelo de Deep Learning para diferenciar entre imagens boas e imagens com defeitos (como grandes quebras, pequenas rachaduras ou contaminação).

### Por que Aprendizado Supervisionado? (Fase Inicial)
Nesta primeira abordagem (o script `app_v2.py`), utilizamos o **Aprendizado Supervisionado**. Isso significa que durante a fase de treinamento, nós dizemos ativamente ao modelo: *"Olhe, esta imagem é uma garrafa NORMAL (Rótulo 0), e esta imagem é uma garrafa DEFEITUOSA (Rótulo 1)"*. O modelo aprende a associar os padrões visuais aos rótulos que nós fornecemos.

### Por que Redes Neurais Convolucionais (CNN)?
Uma imagem colorida de 128x128 pixels tem 49.152 valores numéricos (128x128x3 canais de cor). Se usássemos uma rede neural tradicional (densa/MLP), teríamos milhões de pesos para treinar, o que seria lento e propenso ao *overfitting* (quando o modelo decora os dados de treino mas erra em dados novos). 

As **CNNs** (Redes Neurais Convolucionais) resolvem isso porque:
1. **Compartilhamento de Pesos:** Elas usam "filtros" (kernels) que deslizam pela imagem procurando características (como uma linha reta, uma curva, uma borda), o que reduz drasticamente o número de parâmetros.
2. **Invariância a Translação:** Se um defeito estiver no canto superior esquerdo ou no canto inferior direito, a convolução o encontrará do mesmo jeito.

---

## 2. Análise do Código: Linha a Linha (`app_v2.py`)

Abaixo, dissecamos o código do nosso classificador supervisionado.

### Importação de Bibliotecas
```python
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
```
* **Numpy:** Fundamental para manipular as imagens como matrizes matemáticas multidimensionais.
* **Matplotlib / Seaborn:** Usados para desenhar gráficos (como a Curva ROC, matriz de confusão e visualização de imagens).
* **OS / PIL (Pillow):** Usados para navegar nas pastas do computador e carregar as imagens do disco.
* **TensorFlow / Keras:** A nossa "caixa de ferramentas" de Inteligência Artificial. O Keras simplifica a criação de redes neurais empilhando camadas.

### Fase 1: Configurações Iniciais
```python
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15
```
* **Tamanho da Imagem:** Redimensionamos todas as imagens para 128x128. Imagens maiores exigem mais memória e poder computacional.
* **BATCH_SIZE (32):** O modelo não olha para todas as imagens de uma vez. Ele processa lotes de 32 imagens, calcula o erro e ajusta os pesos.
* **EPOCHS (15):** Quantas vezes o modelo vai ver TODO o conjunto de dados de treino. Se for pouco, ele não aprende o suficiente (underfitting); se for muito, ele pode memorizar (overfitting).

### Fase 2: Carregamento dos Dados
```python
def load_images_from_folder(folder_path, label, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    # ... inicializa listas vazias ...
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img = img.resize(img_size)
            img = np.array(img) / 255.0  # Normalização
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)
```
Nesta etapa crucial, definimos a função que carrega as imagens do disco para a memória RAM. 
* **O Pulo do Gato (`img / 255.0`):** Os pixels das imagens originalmente têm valores de 0 a 255. Redes Neurais odeiam números grandes; eles causam instabilidade no cálculo dos gradientes (matemática por trás do treino). Por isso, dividimos por 255, normalizando tudo para valores entre 0 e 1.

O script carrega imagens normais como `label=0` e categorias como `broken_large`, `broken_small` e `contamination` como `label=1` (defeitos).

### Fase 3: Divisão Treino / Validação
```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)
```
* Por que separar? Precisamos avaliar se o modelo está realmente a aprender ou apenas a decorar os dados. Por isso, pegamos 20% (`test_size=0.2`) dos dados de treino e "escondemos" do modelo para usar apenas na validação ao final de cada época.

### Fase 5: Construção da Arquitetura CNN
```python
model = Sequential(name="CNN_Defect_Detector")
model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
```
Aqui instanciamos o modelo sequencial. A imagem entra com 128x128 pixels e 3 canais (Red, Green, Blue).

```python
# Bloco 1: Características básicas
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))  # 128x128 -> 64x64
```
* **Conv2D(32):** O modelo usa 32 "filtros" de 3x3 pixels. Nesta camada inicial, estes filtros aprenderão a reconhecer bordas, cantos e cores. O `activation='relu'` garante que valores negativos fiquem em zero, introduzindo não-linearidade (permitindo ao modelo aprender padrões complexos, caso contrário ele seria só uma equação de reta gigante).
* **MaxPool2D:** É uma técnica de "resumo". Ele pega um quadrado 2x2 de pixels e mantém apenas o maior valor. Isso reduz a imagem pela metade (128 para 64), diminuindo o processamento mas preservando a característica principal.

Isto se repete mais duas vezes, aumentando os filtros (64, depois 128) e reduzindo o tamanho (32x32, depois 16x16). À medida que a rede fica mais profunda, ela deixa de ver "linhas e cantos" para ver características complexas (ex: a textura inteira de uma anomalia).

```python
# Classificador
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularização
model.add(Dense(1, activation='sigmoid'))  # Saída binária
```
* **Flatten:** A rede neural final não aceita matrizes 2D. O Flatten "achata" a matriz 16x16x128 num único vetor gigante 1D.
* **Dense(128):** Uma rede neural tradicional totalmente conectada que combina todas as características extraídas para tomar uma decisão.
* **Dropout(0.5):** Uma técnica brilhante que "desliga" aleatoriamente 50% dos neurónios durante o treino. Isso impede que um único neurónio fique sobrecarregado de responsabilidade, forçando toda a rede a aprender de forma mais robusta (prevenindo o overfitting).
* **Dense(1, sigmoid):** A camada de saída tem apenas 1 neurónio. A função **Sigmoid** esmaga o resultado matemático entre 0 e 1, que interpretamos como probabilidade (ex: 0.85 = 85% de probabilidade de ter defeito).

### Fase 6: Compilação do Modelo
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```
* **Optimizer ('adam'):** É o motor do modelo. Ele usa uma matemática avançada (Gradient Descent com Momentum adaptativo) para ajustar os pesos da rede e torná-la cada vez mais precisa.
* **Loss ('binary_crossentropy'):** A "bússola" do erro. Esta função de custo calcula quão errada foi a previsão do modelo em comparação à realidade, penalizando fortemente previsões muito confiantes que estão erradas.
* **Metrics:** Especificamos que além da "exatidão" (accuracy), queremos avaliar precisão e recall (muito importante na indústria).

### Fase 7: Treinamento
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
```
Nesta etapa, a mágica acontece (Backpropagation). O modelo olha as imagens num sentido (Forward Pass), erra, vê o erro, e atualiza de trás para frente os seus pesos de modo a errar menos da próxima vez. Isso é repetido `EPOCHS` vezes.

### Fases Finais: Avaliação e Matriz de Confusão
Ao fim do treino, testamos o modelo em imagens de teste (`test_images`) que ele **nunca viu na vida**.

A **Matriz de Confusão** e as Métricas ajudam a entender:
* **Falsos Positivos (FP):** O modelo disse que tinha defeito, mas estava normal. É o "alarme falso". Custa tempo na inspeção de qualidade, mas é tolerável.
* **Falsos Negativos (FN):** O modelo disse que estava normal, mas Tinha defeito. Este é o erro crítico! Significa enviar para o cliente uma garrafa quebrada. Na indústria, queremos o *Recall* lá em cima para minimizar os FN.
* **Curva ROC e AUC:** Mede a capacidade geral do modelo de distinguir entre as classes. Uma área (AUC) de 0.50 significa que o modelo chuta aleatoriamente. Mais próximo de 1.00 significa um classificador perfeito.

---

## Próximos Passos (A Limitação da Abordagem Atual)
O problema do Aprendizado Supervisionado na indústria é que os defeitos são extremamente raros. Muitas vezes temos milhões de fotos de garrafas boas, e quase nenhuma foto de garrafa defeituosa para treinar. E mais: e se aparecer um tipo novo de defeito amanhã que o modelo nunca viu?

Por isso, na **Parte 2 do nosso tutorial**, entraremos no mundo do **Aprendizado Não-Supervisionado com Autoencoders**, onde treinaremos o modelo APENAS com imagens normais!

***Siga para o arquivo TUTORIAL_02_AUTOENCODER.md***
