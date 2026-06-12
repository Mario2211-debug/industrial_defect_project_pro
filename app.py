import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input
from keras.models import Model
from keras.preprocessing import image_dataset_from_directory

# =========================
# 1. DATASET (MVTec AD)
# =========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_path = "dataset/bottle/train/good"
test_path  = "dataset/bottle/test/broken_small"

train_ds = image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None
)

test_ds = image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=1,
    label_mode=None
)

# normalização
train_ds = train_ds.map(lambda x: x / 255.0)
test_ds = test_ds.map(lambda x: x / 255.0)

# =========================
# 2. AUTOENCODER (CNN)
# =========================
input_img = Input(shape=(128, 128, 3))

# Encoder
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPool2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPool2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
encoded = MaxPool2D((2,2))(x)

# Decoder
x = Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)

x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)

output = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

model = Model(input_img, output)

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# =========================
# 3. TREINO (SÓ NORMAL)
# =========================
model.fit(
    train_ds,
    epochs=15
)

# =========================
# 4. THRESHOLD DE ANOMALIA
# =========================
def anomaly_score(img):
    recon = model.predict(img[None, ...], verbose=0)[0]
    return np.mean((img - recon) ** 2)

# calcular threshold usando treino
scores = []

for batch in train_ds.take(10):
    for img in batch:
        scores.append(anomaly_score(img.numpy()))

threshold = np.mean(scores) + 2 * np.std(scores)

print("\nThreshold:", threshold)

# =========================
# 5. TESTE
# =========================
print("\n--- TESTE ---")

for i, batch in enumerate(test_ds.take(10)):
    img = batch[0].numpy()

    score = anomaly_score(img)

    if score > threshold:
        label = "DEFECT"
    else:
        label = "NORMAL"

    print(f"Image {i} | Score: {score:.5f} | {label}")

    # mostrar imagem + erro
    recon = model.predict(img[None, ...], verbose=0)[0]
    error_map = np.mean((img - recon) ** 2, axis=-1)

    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.title("Error Map")
    plt.imshow(error_map, cmap='hot')

    plt.show()