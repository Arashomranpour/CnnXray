<div align="center">

# 🩻 Chest X-Ray CNN Classifier

**A binary convolutional neural network for chest X-ray images, trained with data augmentation, class weights and smart callbacks.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

</div>

---

## ✨ Overview

`cnnxray.ipynb` trains a CNN on grayscale chest X-rays:

- 🖼️ Images are loaded from `data/` (train) and `test_data/` (validation) with Keras `ImageDataGenerator`, resized to **500 × 500** grayscale, with rescaling, shear and zoom augmentation.
- ⚖️ **Balanced class weights** compensate for class imbalance.
- 🧠 `Sequential` CNN with stacked `Conv2D` / `MaxPool2D` layers and a sigmoid output (`binary_crossentropy`, Adam).
- 🛑 Callbacks: `EarlyStopping`, `ReduceLROnPlateau` and `ModelCheckpoint` (best weights saved as `chestxray.h5`).
- 📈 Training/validation accuracy and loss curves are plotted at the end.

> ⚠️ Educational project - not a medical device.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/CnnXray.git
cd CnnXray
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
jupyter notebook cnnxray.ipynb
```

Prepare the data in two folders, each containing one sub-folder per class (for example the Kaggle *Chest X-Ray Images (Pneumonia)* dataset):

```
data/        # training images
test_data/   # validation images
```

## 📁 Project Structure

```
.
└── cnnxray.ipynb
```

## 🛠️ Tech Stack

`TensorFlow` · `Keras` · `scikit-learn` · `NumPy` · `Matplotlib`
