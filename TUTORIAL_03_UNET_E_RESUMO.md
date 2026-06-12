# Tutorial Parte 3: Segmentação a Nível de Pixel e Resumo Geral

Nos passos anteriores, respondemos a duas grandes perguntas:
1. **"Tem defeito?"** -> Classificação com CNN ([Tutorial 1](TUTORIAL_01_FUNDAMENTOS_E_CNN.md))
2. **"Tem algo anômalo na garrafa inteira?"** -> Detecção de Anomalias com Autoencoder ([Tutorial 2](TUTORIAL_02_AUTOENCODER.md))

Porém, na indústria moderna (como na soldagem a laser de peças minúsculas de carros), não basta saber que a peça está com defeito. O braço robótico de reparo precisa saber **"Exatamente QUAIS PIXELS constituem o defeito?"**.

Para resolver isso, entramos no mundo da **Segmentação Semântica a Nível de Pixel**, utilizando a famosa rede **U-Net**.

---

## 1. O que é Segmentação e a U-Net?

A Segmentação é a tarefa de classificar *cada pixel individualmente*. Se a imagem tem 128x128 pixels, a rede fará 16.384 previsões (uma para cada pixel, dizendo se ele é Normal ou Defeito).

A **U-Net** foi originalmente criada em 2015 para descobrir câncer em imagens médicas. Ela tem um formato que lembra a letra "U":
* Ela **desce (Contração/Encoder):** Extrai as características da imagem (bordas, formas) mas vai perdendo a exata "localização espacial" (onde as coisas estão).
* Ela **sobe (Expansão/Decoder):** Tenta recuperar a localização para pintar a imagem final.
* **O Pulo do Gato (Skip Connections):** A U-Net copia imagens inteiras do lado esquerdo (descida) e "cola" no lado direito (subida). Isso permite que a rede combine a inteligência abstrata de "o que é um defeito" com a exatidão visual de "onde ele estava na foto original".

---

## 2. Análise do Código (`app_v4_unet.py`)

Abra o script `app_v4_unet.py` para acompanhar.

### Carregando Máscaras (O Gabarito)
```python
mask = Image.open(mask_path).convert('L').resize(img_size)
mask_array = (np.array(mask) > 0).astype(np.float32)
```
Diferente das fases anteriores, não basta um "Rótulo 1 = Defeito". Agora, para cada imagem com defeito, nós carregamos do dataset MVTec AD uma **Máscara (Ground Truth)**. A máscara é uma imagem preta onde apenas a rachadura exata está pintada de branco. Isso será o "professor" da nossa rede neural.

### A Arquitetura U-Net
```python
# ENCODER
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)
```
Este bloco repete-se várias vezes. As convoluções (`Conv2D`) encontram padrões. O `MaxPooling2D` reduz o tamanho pela metade.

```python
# DECODER e SKIP CONNECTION
u7 = UpSampling2D((2, 2))(c6)
u7 = concatenate([u7, c1]) # <-- Mágica! Junta com a camada c1 original
c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
```
Na subida, o `UpSampling2D` dobra o tamanho da imagem. O `concatenate` pega o detalhe visual exato da camada `c1` (antes do primeiro MaxPooling) e cruza com a informação atual. Sem esse `concatenate`, a segmentação ficaria um borrão horrível!

### Métricas de Segmentação (Dice e IoU)
Na Fase 1 usamos *Acurácia*. Na U-Net, acurácia não serve. Imagine uma imagem enorme, toda preta, com 1 pixel branco de defeito. Se a rede pintar tudo de preto, ela terá 99,9% de acurácia, mas errou 100% do defeito!

Por isso, usamos:
```python
iou_score = np.sum(intersection) / np.sum(union)
dice_score = 2 * np.sum(intersection) / (np.sum(preds_bin) + np.sum(y_all))
```
* **IoU (Intersection over Union):** "Área em que a rede acertou" dividida pela "Área que a rede pintou + Área do defeito real".
* **Dice Score:** Semelhante ao F1-Score, mede a sobreposição exata. 1.0 é a perfeição, 0.0 é erro total. Na indústria médica e industrial, um Dice > 0.7 já é excelente.

---

## 3. Resumo Final do Projeto e Arquiteturas

Neste projeto de Inspeção Visual com Visão por Computador (Dataset MVTec AD Bottle), abordamos três grandes pilares do Deep Learning moderno. Qual você deve escolher para o seu projeto real na empresa?

### 1. CNN Clássica Supervisioada (`app_v2.py`)
* **Uso:** Você quer apenas um sinal "Verde" ou "Vermelho" na linha de produção e possui **MILHARES** de fotos de produtos defeituosos.
* **Pró:** Muito rápido de inferir, altamente preciso nas classes que foi treinado.
* **Contra:** Se a fábrica criar um defeito novo amanhã (ex: respingo de solda, que ela nunca viu no treino), o modelo não saberá o que é.

### 2. Autoencoder Não-Supervisionado (`app_v3_autoencoder.py`)
* **Uso:** Você **NÃO** tem fotos de peças defeituosas. A linha é tão boa que só saem peças perfeitas.
* **Pró:** Consegue detectar **QUALQUER anomalia**, mesmo um defeito que o humano nunca imaginou. Só precisa de fotos de coisas normais para treinar.
* **Contra:** Mais difícil de treinar. Os mapas de calor dão a região da anomalia, mas as bordas do defeito não são nítidas e pode gerar falsos positivos se a iluminação da fábrica variar um pouco.

### 3. Segmentação Semântica U-Net (`app_v4_unet.py`)
* **Uso:** O braço robótico precisa ir com um laser exatamente nos pixels do defeito para o reparar. Você quer contornos cirúrgicos.
* **Pró:** O suprassumo da detecção. Precisão cirúrgica de localização espacial.
* **Contra:** É o mais caro para treinar e preparar os dados. Exige que um ser humano desenhe máscaras ("pinte os defeitos") à mão em centenas de imagens no Photoshop para treinar a rede.

### Conclusão
O objetivo da disciplina e do trabalho foi amplamente atingido. Começamos com classificações simples e evoluímos até a ponta da tecnologia moderna de segmentação de imagens industriais!
