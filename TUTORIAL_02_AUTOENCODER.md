# Tutorial Parte 2: Aprendizado Não-Supervisionado com Autoencoders

No [Tutorial 1](TUTORIAL_01_FUNDAMENTOS_E_CNN.md), nós criamos uma CNN Clássica. Para isso funcionar, nós precisávamos de fotos de garrafas com defeito durante a fase de treino. Mas na vida real da indústria, defeitos são raros. Às vezes, você só tem fotos de produtos perfeitos!

Como ensinamos uma Inteligência Artificial a encontrar um defeito se ela nunca viu um defeito na vida?
A resposta é o **Aprendizado Não-Supervisionado**.

---

## 1. O que é um Autoencoder?

Imagine que eu peço para você memorizar o rosto da Monalisa de Leonardo da Vinci, olhando para ela por horas a fio. Você decora cada traço, cada sorriso enigmático.
Amanhã, eu te mostro uma réplica da Monalisa, mas alguém desenhou um bigode nela com caneta preta.
Mesmo sem eu nunca ter te ensinado "o que é um bigode", você imediatamente aponta para o bigode e diz: *"Isso está errado, isso não faz parte da Monalisa original"*.

É exatamente isso que um **Autoencoder** faz.

Um Autoencoder é uma Rede Neural dividida em duas partes:
1. **Encoder (Codificador):** Pega na imagem original (128x128) e vai comprimindo ela até virar um vetor minúsculo (chamado de espaço latente). É como compactar um arquivo `.zip`.
2. **Decoder (Decodificador):** Pega nesse `.zip` minúsculo e tenta "desenhar" a imagem original de volta, nas mesmas dimensões (128x128).

O truque mestre deste projeto está aqui: **Nós treinamos o Autoencoder passando APENAS imagens de garrafas PERFEITAS (normais).**

### O Efeito Prático na Inspeção
* Se você der uma garrafa normal ao Autoencoder treinado, ele a comprime e descomprime com facilidade. A imagem de entrada e a imagem de saída ficam quase idênticas.
* Se você der uma garrafa **com rachadura**, o Autoencoder vai olhar, comprimir, e na hora de descomprimir ele vai pensar: *"Eu só sei desenhar garrafas perfeitas!"* e vai **desenhar uma garrafa perfeita por cima da rachadura**.
* Quando você subtrai matematicamente a imagem original (que tinha a rachadura) pela imagem de saída (que o Autoencoder desenhou sem rachadura), a rachadura "acende" como um pisca-alerta! Acabamos de descobrir o defeito sem nunca ter ensinado o que era um defeito.

---

## 2. Análise do Código (`app_v3_autoencoder.py`)

Abra o script `app_v3_autoencoder.py` para acompanhar.

### Carregando os Dados (O Truque)
```python
# O grande truque: O treino só vê imagens sem defeito!
X_train = load_images(os.path.join(TRAIN_PATH, "good"))
```
Diferente da Parte 1, onde carregávamos imagens `good` (0) e `broken` (1) para treinar, aqui carregamos **apenas as pastas `good`**. A rede não faz a menor ideia do que seja uma garrafa partida.

### A Arquitetura do Autoencoder
```python
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# ENCODER (Comprime)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x) # Cai pela metade (64x64)
# ... mais convoluções e poolings ...
encoded = MaxPooling2D((2, 2), padding='same')(x) # Espaço latente (16x16)
```
O Encoder vai extraindo as características mais fundamentais de uma garrafa boa e jogando o resto fora.

```python
# DECODER (Reconstrói)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x) # Dobra o tamanho (32x32)
# ... mais convoluções e upsamplings ...
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # Imagem final (128x128)
```
O Decoder usa o `UpSampling2D` (o oposto do MaxPooling, ele dobra o tamanho da matriz) e tenta reescrever a imagem original a partir daquela memória comprimida. A última camada tem 3 canais (RGB) e usa `sigmoid` para garantir que os pixels fiquem entre 0 e 1, igual à imagem que entrou.

### O Treinamento Incomum
```python
history = autoencoder.fit(X_train, X_train, epochs=EPOCHS, ...)
```
Olhe bem para os parâmetros do `fit`: `X_train, X_train`. Nós estamos passando a imagem de entrada como sendo **o objetivo final (target)**. A rede calcula o erro (Loss, usando Erro Quadrático Médio - MSE) entre o que ela gerou e a imagem original. O objetivo dela é ser um "espelho perfeito" para garrafas boas.

### O Erro de Reconstrução e os Mapas de Calor (Heatmaps)
A verdadeira mágica acontece na hora do teste:
```python
# O Autoencoder tenta reconstruir as imagens de teste
reconstructed = autoencoder.predict(X_test)

# O "erro" é a diferença entre a original e a reconstruída
mse_maps = np.mean(np.square(X_test - reconstructed), axis=-1)
```
Aqui nós testamos com imagens que contêm rachaduras (que a rede nunca viu).
O `np.square(X_test - reconstructed)` compara a garrafa defeituosa com a garrafa perfeita que o Autoencoder gerou, pixel por pixel. 

Se um pixel na garrafa original era Preto (rachadura) e o Autoencoder gerou Transparente (vidro normal), a diferença (o erro) nesse pixel será ENORME.
Esse mapa de erros `mse_maps` é um **Heatmap (Mapa de Calor)**. Ele não só nos diz que a garrafa tem defeito, mas ele aponta exatamente **onde** o defeito está!

### Métrica de Avaliação (Image-Level AUROC)
```python
anomaly_scores = np.max(mse_maps, axis=(1, 2))
fpr, tpr, thresholds = roc_curve(y_test, anomaly_scores)
roc_auc = auc(fpr, tpr)
```
Como decidimos se a garrafa vai pro lixo ou não? Nós pegamos o mapa de calor e procuramos o "pixel mais errado" de todos (`np.max`). Se esse erro máximo for maior que um certo limiar, classificamos a garrafa como defeituosa. A curva ROC (e a AUC) avalia o quão bem esse escore máximo separa as garrafas boas das ruins.

---

## O Limite do Autoencoder
Autoencoders são maravilhosos para detecção de anomalias, mas as reconstruções costumam ser um pouco embaçadas (blurry). Isso pode gerar pequenos erros nas bordas das garrafas, criando alarmes falsos. Além disso, nós conseguimos descobrir mais ou menos onde o defeito está com o Mapa de Calor, mas os contornos do defeito não ficam muito exatos.

Se precisarmos de **precisão cirúrgica** para saber exatamente qual pixel é defeito e qual não é (Segmentação a Nível de Pixel), precisamos usar uma arquitetura mais avançada, focada em segmentação.

Na **Parte 3 do tutorial**, resolveremos este problema criando uma **U-Net** para segmentação semântica, comparando as imagens com o "Gabarito" (`ground_truth`) fornecido pelo Dataset!

***Siga para o arquivo TUTORIAL_03_UNET_E_RESUMO.md***
