# Efficient Deep Learning Approach for Diagnosis of ADHD in Children Based on EEG Signals

Classifying children with **Attention-Deficit/Hyperactivity Disorder (ADHD)** versus a healthy control group from **EEG signals**, by turning each EEG epoch into a Morlet time-frequency RGB image and training convolutional and residual neural networks. This repository accompanies the paper published in *Cognitive Computation* (2024).

> ADHD is a common behavioral disorder in children that can persist into adulthood if untreated, so early and reliable diagnosis matters. This work explores whether an image-based deep learning pipeline applied to EEG time-frequency representations can support that diagnosis.

---

## Overview

The pipeline takes raw EEG recordings from two groups of children (an ADHD group and a control group), slices them into short overlapping epochs, and converts each epoch into a colour (RGB) time-frequency image using a **Morlet wavelet** transform. These images are then used to train two classifiers:

1. A **convolutional neural network (CNN)** with dropout regularisation.
2. A **customised residual network (ResNet)** built from scratch with identity and convolutional residual blocks.

Both models are evaluated with **10-fold stratified cross-validation**, at two levels of granularity:

- **Epoch (segment) level** — each EEG segment is classified independently.
- **Subject level** — predictions are aggregated per child to produce a per-subject diagnosis.

---

## What's inside

| File | Description |
|------|-------------|
| `ADHD.ipynb` | End-to-end Google Colab notebook: data loading, Morlet RGB image generation, CNN and ResNet model definitions, training, and 10-fold cross-validation. |
| `README.md` | This file. |

The notebook expects the EEG dataset to be available on Google Drive under `adhd_dataset/` (with `ADHD_part*` and `Control_part*` files in EEGLAB `.set` format). The dataset itself is **not** included in this repository.

---

## Methods / Approach

**Signal preprocessing & feature representation**
- EEG recordings are read with [MNE-Python](https://mne.tools/) (`mne.io.read_raw_eeglab`).
- Each recording is segmented into **2-second epochs with 1.5 s overlap** (`mne.make_fixed_length_epochs`).
- Each epoch is transformed into a time-frequency representation with a **Morlet wavelet** (`mne.time_frequency.tfr_morlet`, `n_cycles=5`) and rendered as an **RGB image**, which becomes the model input.

**Models**
- **CNN with dropout** — a stack of `Conv2D` / `MaxPooling2D` layers with dropout for regularisation.
- **Customised ResNet** — a residual network built from scratch using custom `identity_block` and `convolutional_block` functions (the notebook also imports `ResNet50V2` from Keras for reference/comparison).
- Both models use the **Adam** optimiser and **binary cross-entropy** loss.

**Evaluation**
- **10-fold stratified cross-validation** (`StratifiedKFold(n_splits=10, shuffle=True)`) at both the epoch level and the subject level.
- Reported metrics include accuracy, precision, recall, and F1-score, plus confusion matrices (`sklearn.metrics`).

---

## How to run

This project is a self-contained Google Colab notebook.

1. Open `ADHD.ipynb` in [Google Colab](https://colab.research.google.com/) (GPU runtime recommended).
2. Place the EEG dataset on your Google Drive under `MyDrive/adhd_dataset/` (EEGLAB `.set` files for the `ADHD_part*` and `Control_part*` groups).
3. Run the cells top to bottom. The notebook mounts Drive, installs dependencies, generates the Morlet RGB images, and trains/evaluates the models.

The notebook installs its extra dependency inline:

```bash
pip install pandasql==0.7.1
```

Core libraries used (already available in Colab):

```text
tensorflow / keras   # CNN and ResNet models
mne                  # EEG I/O, epoching, Morlet time-frequency transform
numpy, scipy, pandas # numerical and data handling
scikit-learn         # cross-validation and metrics
matplotlib           # RGB image rendering and plots
```

> Note: `ADHD.ipynb` imports `resnets_utils` (helper functions such as `convert_to_one_hot`). This utility module is a standard companion file used by the ResNet code and is not included in the repository; supply it (or adapt the helpers inline) when running outside the original environment.

---

## Results

The accompanying *Cognitive Computation* (2024) paper reports the following cross-validated results on the dataset of **61 children with ADHD** and **60 healthy controls**:

| Model | Level | Accuracy | F1-score |
|-------|-------|----------|----------|
| CNN | Epoch / segment | 92.52% | 93.6% |
| Customised ResNet | Epoch / segment | **96.8%** | **97.1%** |
| CNN | Subject | 96.5% | — |
| Customised ResNet | Subject | **98.6%** | — |

The customised ResNet outperforms the CNN on accuracy, precision, recall, and F1-score.

> **Caveat (from the authors):** the data were augmented and the study is based on a single experiment with a relatively small number of children. The reported accuracies are computed over augmented epoch samples derived from this cohort, so broader validation across a larger and more diverse population is needed before clinical conclusions are drawn.

---

## Tech stack

**Python · TensorFlow / Keras · MNE-Python · scikit-learn · NumPy / SciPy / pandas · Matplotlib · Google Colab**

---

## Citation

If you use this work, please cite the associated paper:

> *Efficient Deep Learning Approach for Diagnosis of ADHD in Children Based on EEG Signals.* Cognitive Computation, 2024.

---

*Maintained by [Hamid Jahani](https://github.com/sheperd007).*
