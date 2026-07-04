# Industrial Defect Detection

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

Automatic visual inspection of industrial products using deep learning, built as a Computer Vision course project on the **MVTec AD** dataset (`bottle` category). The project is split into four phases, each tackling the same problem — *is this product defective?* — with a different technique, from plain supervised classification to unsupervised anomaly segmentation and transfer learning.

> 🇵🇹 Versão em português mais abaixo / Portuguese version further below.

## Project phases

| Phase | Script | Technique | What it answers |
|---|---|---|---|
| 1 | `app.py` / `app_v2.py` | CNN (supervised classification) | Is the product normal or defective? |
| 2 | `app_v3_autoencoder.py` | Convolutional Autoencoder (unsupervised) | How different is this image from a "normal" reconstruction? |
| 3 | `app_v4_unet.py` | U-Net (pixel-level segmentation) | *Where* exactly is the defect? |
| 4 | `app_v5_resnet.py` | ResNet50 (transfer learning / feature extraction) | Can pretrained ImageNet features separate good from defective? |

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — `bottle` category:

```
train/good/          # normal images only, used for training
test/good/            # normal images, used for evaluation
test/broken_large/    # defective images, used for evaluation
```

The dataset itself is not committed to this repo (see `.gitignore`) — download it from the link above and place it locally before running any script.

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `tensorflow`, `matplotlib`, `seaborn`, `scikit-learn`.

Run any phase directly, e.g.:

```bash
python app_v3_autoencoder.py
```

## Tutorials

Step-by-step write-ups of the theory behind each phase:

- [`TUTORIAL_01_FUNDAMENTOS_E_CNN.md`](TUTORIAL_01_FUNDAMENTOS_E_CNN.md) — CNN fundamentals and supervised classification
- [`TUTORIAL_02_AUTOENCODER.md`](TUTORIAL_02_AUTOENCODER.md) — autoencoders and unsupervised anomaly detection
- [`TUTORIAL_03_UNET_E_RESUMO.md`](TUTORIAL_03_UNET_E_RESUMO.md) — U-Net segmentation and phase summary

`REPORT_TEMPLATE.md` contains the template used to report metrics (Image AUROC, Pixel AUROC, Precision, Recall, Dice Score) per phase.

---

## 🇵🇹 Versão em português

Inspeção visual automática de produtos industriais com deep learning, feita como projeto da disciplina de Visão por Computador sobre o dataset **MVTec AD** (categoria `bottle`). O projeto está dividido em quatro fases, cada uma a resolver o mesmo problema — *este produto tem defeito?* — com uma técnica diferente, desde classificação supervisionada simples até segmentação de anomalias não-supervisionada e transfer learning.

### Fases do projeto

| Fase | Script | Técnica | O que responde |
|---|---|---|---|
| 1 | `app.py` / `app_v2.py` | CNN (classificação supervisionada) | O produto é normal ou tem defeito? |
| 2 | `app_v3_autoencoder.py` | Autoencoder convolucional (não-supervisionado) | Quão diferente é esta imagem de uma reconstrução "normal"? |
| 3 | `app_v4_unet.py` | U-Net (segmentação ao nível do pixel) | *Onde* exatamente está o defeito? |
| 4 | `app_v5_resnet.py` | ResNet50 (transfer learning / extração de features) | As features pré-treinadas na ImageNet separam bem produtos bons de defeituosos? |

### Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — categoria `bottle`:

```
train/good/          # apenas imagens normais, usadas no treino
test/good/            # imagens normais, usadas na avaliação
test/broken_large/    # imagens com defeito, usadas na avaliação
```

O dataset não está incluído no repositório (ver `.gitignore`) — faz o download no link acima e coloca-o localmente antes de correr qualquer script.

### Instalação

```bash
pip install -r requirements.txt
```

Requisitos: `tensorflow`, `matplotlib`, `seaborn`, `scikit-learn`.

Corre qualquer fase diretamente, por exemplo:

```bash
python app_v3_autoencoder.py
```

### Tutoriais

Explicações passo-a-passo da teoria por trás de cada fase:

- [`TUTORIAL_01_FUNDAMENTOS_E_CNN.md`](TUTORIAL_01_FUNDAMENTOS_E_CNN.md) — fundamentos de CNN e classificação supervisionada
- [`TUTORIAL_02_AUTOENCODER.md`](TUTORIAL_02_AUTOENCODER.md) — autoencoders e deteção de anomalias não-supervisionada
- [`TUTORIAL_03_UNET_E_RESUMO.md`](TUTORIAL_03_UNET_E_RESUMO.md) — segmentação com U-Net e resumo das fases

`REPORT_TEMPLATE.md` contém o modelo usado para reportar as métricas (Image AUROC, Pixel AUROC, Precision, Recall, Dice Score) de cada fase.
