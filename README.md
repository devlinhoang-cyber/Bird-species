# Bird Species classification using Convolutional Neural Networks
Devlin Hoang| Practical Homework 3 — Deep Learning  
---

## Overview

This project builds CNN models to classify North American birds from audio recordings. Bird calls got converted into mel spectrogram images, then used them as input for CNN classifiers. Two models were developed: a binary classifier and a multi-class classifier. 
---

## Dataset

Using audio spectrograms in an HDF5 archive (`bird_spectrograms.hdf5`) containing mel spectrograms for 12 species.

---

## Repository Structure

```
├── Practical-homework3.ipynb     # Main notebook with all code and results
├── Bird data/
│   ├── bird_spectrograms.hdf5    # HDF5 spectrogram dataset
│   ├── test1.mp3                 # External test clip 1
│   ├── test2.mp3                 # External test clip 2
│   └── test3.mp3                 # External test clip 3
├── bird_metadata.csv             # Metadata for all 1,981 samples
├── train_extended.csv            # Extended training metadata
└── README.md
```

---

## Preprocessing

- Spectrograms are stored as `(frequency, time, samples)` in HDF5, then got reshaped `(samples, 128, 517, 1)` to fit CNN models
- Normalized pixel values to `[0, 1]` by dividing by 255
- Labels converted to integers with `LabelEncoder` and one-hot encoded for multi-class training
- 70/30 stratified train/test split

---

## Models

### Binary Classifier — American Crow vs. Song Sparrow

| Conv2D (16 filters, 3×3, ReLU) | (128, 517, 16) | 160 |
| MaxPooling2D (2×2) | (64, 258, 16) | — |
| Conv2D (32 filters, 3×3, ReLU) | (64, 258, 32) | 4,640 |
| MaxPooling2D (2×2) | (32, 129, 32) | — |
| Flatten | (132,096) | — |
| Dropout (0.2) | (132,096) | — |
| Dense (32, ReLU) | (32) | 4,227,104 |
| Dense (1, sigmoid) | (1) | 33 |

- **Optimizer:** RMSprop | **Loss:** Binary Cross-Entropy | **Epochs:** 20 | **Batch size:** 30

### Multi-Class Classifier — 12 Species

| Conv2D (32 filters, 3×3, ReLU) | (128, 517, 32) | 320 |
| MaxPooling2D (2×2) | (64, 258, 32) | — |
| Conv2D (64 filters, 3×3, ReLU) | (64, 258, 64) | 18,496 |
| MaxPooling2D (2×2) | (32, 129, 64) | — |
| Conv2D (128 filters, 3×3, ReLU) | (32, 129, 128) | 73,856 |
| MaxPooling2D (2×2) | (16, 64, 128) | — |
| Conv2D (256 filters, 3×3, ReLU) | (16, 64, 256) | 295,168 |
| MaxPooling2D (2×2) | (8, 32, 256) | — |
| Flatten | (65,536) | — |
| Dropout (0.3) | (65,536) | — |
| Dense (128, ReLU) | (128) | 8,388,736 |
| Dense (12, softmax) | (12) | 1,548 |

- **Optimizer:** Adam | **Loss:** Categorical Cross-Entropy | **Epochs:** 30 | **Batch size:** 64

---

## Results

### Binary Model (amecro vs. sonspa)

| Training Accuracy | 97.4% |
| Test Accuracy | 89.9% |

### Multi-Class Model (12 species)

| Training Accuracy | 89.7% |
| Test Accuracy | 43.4% |
| Total Parameters | 8,778,124 |

### External Test Clip Predictions

| test1.mp3 | `sonspa` | 34.3% | Song Sparrow |
| test2.mp3 | `whcspa` | 86% | White crowned sparrow |
| test3.mp3 | `whcspa` | 99% | White crowned sparrow |


## How to Run

1. Clone the repository and place the `Bird data/` folder in the project root
2. Open `Practical-homework3.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells in order
4. External MP3 predictions are generated in the final cells

---
