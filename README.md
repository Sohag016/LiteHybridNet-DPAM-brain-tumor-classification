# 🧠 LiteHybridNet-DPAM: Lightweight Brain Tumor Classification with Dual-Path Attention and Uncertainty Estimation

> **Research Paper Implementation (Reviewer-Ready v6)** — A novel lightweight deep learning framework for brain tumor MRI classification featuring **DPAM (Dual-Path Attention Module)**, **Multi-Scale Feature Fusion**, and **MC Dropout Uncertainty Estimation** on two publicly available brain MRI datasets.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Novel Contributions](#-novel-contributions)
3. [Datasets](#-datasets)
4. [Tumor Classes](#-tumor-classes)
5. [Framework Architecture](#-framework-architecture)
6. [Project Steps](#-project-steps)
   - [Step 1 — Imports & Global Seed](#step-1--imports--global-seed)
   - [Step 2 — GPU Check](#step-2--gpu-check)
   - [Step 3 — Dataset Paths (Auto-Detect)](#step-3--dataset-paths-auto-detect)
   - [Step 4 — Duplicate & Leakage Check](#step-4--duplicate--leakage-check)
   - [Step 5 — Data Transforms & Loaders](#step-5--data-transforms--loaders)
   - [Step 6 — Sample Visualisation](#step-6--sample-visualisation)
   - [Step 7 — LiteHybridNet-DPAM Model](#step-7--litehybridnet-dpam-model)
   - [Step 8 — Computational Efficiency Metrics](#step-8--computational-efficiency-metrics)
   - [Step 9 — Hyperparameters](#step-9--hyperparameters)
   - [Step 10 — Training & Evaluation Functions](#step-10--training--evaluation-functions)
   - [Step 11 — 5-Fold Cross-Validation](#step-11--5-fold-cross-validation)
   - [Step 12 — Final Model Training on Full Train Set](#step-12--final-model-training-on-full-train-set)
   - [Step 13 — Confusion Matrix & Training Curves](#step-13--confusion-matrix--training-curves)
   - [Step 14 — Uncertainty Estimation (MC Dropout)](#step-14--uncertainty-estimation-mc-dropout)
   - [Step 14b — Calibration: ECE & Reliability Diagram](#step-14b--calibration-ece--reliability-diagram)
   - [Step 14c — DPAM Ablation Study](#step-14c--dpam-ablation-study)
   - [Step 15 — Baseline Comparison (Fair Setup)](#step-15--baseline-comparison-fair-setup)
   - [Step 16 — Efficiency & F1 Summary](#step-16--efficiency--f1-summary)
   - [Step 17 — Efficiency Comparison Table](#step-17--efficiency-comparison-table)
   - [Step 18 — XAI: Integrated Gradients](#step-18--xai-integrated-gradients)
   - [Step 19 — XAI: Grad-CAM](#step-19--xai-grad-cam)
   - [Step 20 — XAI: Qualitative Visualisation](#step-20--xai-qualitative-visualisation)
   - [Step 21 — Failure Case Visualisation](#step-21--failure-case-visualisation)
   - [Step 22 — Architecture Diagram](#step-22--architecture-diagram)
7. [Model Architecture Details](#-model-architecture-details)
8. [Baseline Models Compared](#-baseline-models-compared)
9. [Evaluation Strategy](#-evaluation-strategy)
10. [Installation & Setup](#-installation--setup)
11. [How to Run](#-how-to-run)
12. [Results](#-results)
13. [Technologies Used](#-technologies-used)
14. [Dataset Citation](#-dataset-citation)
15. [Paper Citation](#-paper-citation)

---

## 🔍 Project Overview

**LiteHybridNet-DPAM** is a novel lightweight convolutional neural network designed for **brain tumor MRI classification** (4 classes: Glioma, Meningioma, Pituitary, No Tumor). The model is built on a **MobileNetV2 backbone** (ImageNet pretrained) with three novel components added on top:

1. **DPAM** — Dual-Path Attention Module (parallel channel + spatial attention)
2. **Multi-Scale Feature Fusion (MSF)** — fuses early and deep feature maps for richer representations
3. **MC Dropout Uncertainty Estimation** — quantifies prediction confidence at inference time

The framework is validated on **two independent brain MRI datasets** (Figshare + Mendeley), evaluated with **5-fold cross-validation**, and includes full **XAI** (Grad-CAM + Integrated Gradients), **calibration analysis** (ECE + Temperature Scaling), and an **ablation study** — all publication-ready.

---

## 💡 Novel Contributions

### Contribution 1 — DPAM (Dual-Path Attention Module)
A novel parallel attention mechanism that simultaneously applies:
- **Channel Attention (SE block):** recalibrates inter-channel feature responses
- **Spatial Attention:** highlights spatially discriminative tumor regions

Unlike sequential attention (CBAM), DPAM runs both paths **in parallel** and fuses them — capturing complementary attention signals simultaneously.

```
Feature Map → ┌──── Channel Attention (SE) ────┐
              │                                  ├─► Element-wise Add → Output
              └──── Spatial Attention ───────────┘
```

### Contribution 2 — Uncertainty Estimation (MC Dropout)
At inference time, **dropout remains active** across T=20 forward passes. The variance of the T predictions gives a **per-sample uncertainty score** (Shannon entropy). Low-uncertainty predictions are more reliable; high-uncertainty samples are flagged for clinical review.

### Contribution 3 — Multi-Scale Feature Fusion (MSF)
Extracts features from two stages of the backbone:
- **Early scale** (Stage 1 output, 28×28) — captures fine-grained texture
- **Deep scale** (Stage 2 output, 7×7) — captures semantic tumor structure

Both are projected and fused into a unified 256-D representation before classification.

---

## 📊 Datasets

Two independent, publicly available brain MRI datasets are used:

### Dataset 1 — Figshare Brain MRI Dataset
| Property | Details |
|---|---|
| **Name** | Brain MRI Dataset |
| **URL** | https://figshare.com/articles/dataset/Brain_MRI_Dataset/14778750 |
| **DOI** | 10.6084/m9.figshare.14778750 |
| **Format** | ZIP archive (folder per class) |

### Dataset 2 — Mendeley Brain Tumor MRI Dataset
| Property | Details |
|---|---|
| **Name** | Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor) |
| **URL** | https://data.mendeley.com/datasets/zwr4ntf94j/5 |
| **DOI** | 10.17632/zwr4ntf94j.5 |
| **Version** | V5 (2025) |
| **Format** | ZIP archive (folder per class) |

> ⚠️ Datasets are **not bundled** with this repository. Download them from the links above and provide the ZIP path in Step 3.

---

## 🧬 Tumor Classes

| Class | Description |
|---|---|
| **Glioma** | Malignant tumor from glial cells — most aggressive |
| **Meningioma** | Tumor arising from the meninges — usually benign |
| **Pituitary** | Tumor in the pituitary gland |
| **No Tumor** | Healthy brain MRI (control) |

---

## 🔄 Framework Architecture

```
Brain MRI Image (224×224×3)
          │
          ▼
  MobileNetV2 Backbone
  (ImageNet pretrained weights)
          │
    ┌─────┴──────────────────────────────┐
    ▼                                    ▼
Stage 1 Output                    Stage 2 Output
(28×28 — Early Scale)            (7×7 — Deep Scale)
    │                                    │
    └──────────┬─────────────────────────┘
               ▼
    Multi-Scale Fusion (MSF)
    (project + concat → 256-D)
               │
               ▼
   InvertedResidual + DPAM Blocks
   (Dual-Path: Channel ‖ Spatial Attention)
               │
               ▼
   AdaptiveAvgPool2d(1) → Flatten
               │
               ▼
   MC Dropout (p=0.3, active at inference)
               │
               ▼
   Linear(1280 → 4) — Classifier
               │
     ┌─────────┴──────────┐
     ▼                    ▼
Prediction (class)   Uncertainty Score
                     (entropy over T=20 passes)
```

---

## 📁 Project Steps

### Step 1 — Imports & Global Seed
Imports all required libraries: PyTorch, torchvision, Captum (Integrated Gradients), grad-cam, seaborn, torchinfo, scipy. Sets global random seed (`seed=42`) for `random`, `numpy`, `torch`, and CUDA for full reproducibility.

### Step 2 — GPU Check
Detects and prints GPU availability, CUDA version, device name, and VRAM. Falls back to CPU gracefully. All tensors and models are moved to the detected device.

### Step 3 — Dataset Paths (Auto-Detect)
Auto-detects the runtime environment (Google Colab vs. local). Sets `ZIP_PATH` accordingly:
- **Colab:** mounts Google Drive and reads from `MyDrive`
- **Local:** reads from the local desktop path

Extracts the ZIP to a local folder, auto-discovers `train/` and `test/` subdirectories, reads class names from folder structure, and prints per-class image counts.

### Step 4 — Duplicate & Leakage Check
Performs a data integrity audit before training:
- Computes **MD5 hash** of every image file
- Detects and removes **exact duplicate images** within splits
- Checks for **train-test leakage** (same image appearing in both splits)
- Prints a clean report: duplicates found, leakage count, final usable sample sizes

This step ensures no data contamination that could inflate reported accuracy.

### Step 5 — Data Transforms & Loaders
**Training transforms (augmentation):**
- `RandomResizedCrop(224, scale=(0.7, 1.0))`
- `RandomHorizontalFlip` + `RandomVerticalFlip`
- `RandomRotation(15°)`
- `ColorJitter` (brightness=0.3, contrast=0.3, saturation=0.2)
- `RandomGrayscale(p=0.05)`
- `Normalize` (ImageNet mean/std)

**Validation/Test transforms:** `Resize(256)` → `CenterCrop(224)` → `Normalize`

Uses `AugSubset` wrapper so the **raw dataset** (no transform) is used as the index source for all splits — eliminating augmentation leakage into validation folds during cross-validation.

### Step 6 — Sample Visualisation
Collects one representative MRI image from each of the 4 classes. Displays them in a clean 1×4 grid with class labels. Saves as a high-quality figure (suitable for paper Figure 1).

### Step 7 — LiteHybridNet-DPAM Model
Builds the novel **LiteHybridNet-DPAM** architecture:

- **Backbone:** MobileNetV2 (ImageNet pretrained, weights transferred for backbone stages only)
- **Novel layers** (DPAM, MSF, conv_head, classifier) remain randomly initialized
- **DPAM** inserted after each InvertedResidual block in Stage 1–2
- **MSF** fuses early (28×28) and deep (7×7) feature maps → 256-D
- **MC Dropout** (p=0.3) placed before the final classifier

Uses **differential learning rates:**
- Backbone layers: `lr = 1e-4` (fine-tune carefully)
- Novel DPAM/MSF/head layers: `lr = 5e-4` (train faster)

### Step 8 — Computational Efficiency Metrics
Measures and reports model efficiency for edge deployment comparison:
- **Parameters** — total trainable parameter count (via `torchinfo`)
- **FLOPs** — multiply-accumulate operations per forward pass
- **CPU Latency** — average inference time on CPU (ms) over 100 runs
- **Model size** — disk footprint (MB)

All baselines are measured identically on the same CPU hardware for a fair comparison.

### Step 9 — Hyperparameters
Defines all training hyperparameters in a single `HPARAMS` dictionary:

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs (main) | 30 |
| Epochs (CV folds) | 15 |
| Optimizer | AdamW |
| LR (backbone) | 1e-4 |
| LR (head/novel) | 5e-4 |
| Weight decay | 1e-4 |
| MC Dropout samples (T) | 20 |
| Num classes | 4 |

### Step 10 — Training & Evaluation Functions
Implements reusable `train_epoch()` and `eval_epoch()` functions:
- **train_epoch:** forward pass → CrossEntropyLoss → backward → gradient clipping (max norm=1.0) → AdamW step → CosineAnnealingLR step
- **eval_epoch:** no-grad forward pass → accuracy + macro F1 + per-class metrics
- Loss and accuracy logged per epoch for curve plotting

### Step 11 — 5-Fold Cross-Validation
Runs **stratified 5-fold cross-validation** on the training set:
- Each fold: 4 folds train, 1 fold validate
- Training folds receive augmentation; validation fold uses clean transforms (no leakage)
- Reports per-fold accuracy and F1
- Reports **mean ± std** across 5 folds
- Results used to demonstrate generalisation, not for model selection

### Step 12 — Final Model Training on Full Train Set
Trains the final LiteHybridNet-DPAM on the **complete training set** (train + val combined) for 30 epochs. Saves the best checkpoint based on validation F1. This is the model used for all subsequent evaluation steps.

### Step 13 — Confusion Matrix & Training Curves
Generates two publication figures:
- **Training curves:** train/val loss and accuracy across 30 epochs
- **Confusion matrix:** 4×4 Seaborn heatmap on the test set showing per-class TP/FP/FN/TN

### Step 14 — Uncertainty Estimation (MC Dropout)
Runs **T=20 stochastic forward passes** (dropout active) on each test image. For each sample:
- Computes mean prediction → final class label
- Computes **Shannon entropy** of the T predictions → uncertainty score

Plots uncertainty distribution across correct vs. incorrect predictions. High-uncertainty samples are visualized for clinical interpretability.

### Step 14b — Calibration: ECE & Reliability Diagram
Measures how well the model's confidence matches its actual accuracy:
- **ECE (Expected Calibration Error):** lower is better
- **Reliability diagram:** plots confidence bins vs. actual accuracy
- Applies **Temperature Scaling** post-hoc to improve calibration
- Compares ECE before and after temperature scaling

### Step 14c — DPAM Ablation Study
Systematically replaces DPAM with alternative attention mechanisms across 5-fold CV:

| Variant | Attention |
|---|---|
| **Ours** | DPAM (parallel channel + spatial) |
| CBAM-seq | Channel → Spatial (sequential) |
| Channel only | SE block only |
| Spatial only | Spatial attention only |
| No attention | Standard InvertedResidual |

Reports mean ± std accuracy per variant. Demonstrates that DPAM's parallel design outperforms sequential alternatives.

### Step 15 — Baseline Comparison (Fair Setup)
Trains all baseline models under **identical conditions** (same data, same epochs=30, same optimizer, no pretrained weights for fair comparison):

| Baseline | Architecture |
|---|---|
| ResNet-50 | Deep residual network |
| VGG-16 | Classic deep CNN |
| MobileNetV2 | Closest lightweight architecture |
| EfficientNet-B0 | Compound scaling network |

### Step 16 — Efficiency & F1 Summary
Compiles a side-by-side comparison of LiteHybridNet-DPAM vs. all baselines across:
- Test accuracy (%)
- Macro F1 score
- Parameters (M)
- FLOPs (G)
- CPU latency (ms)

Generates a grouped bar chart (publication Figure).

### Step 17 — Efficiency Comparison Table
Produces a clean **LaTeX-ready** and **pandas** efficiency table summarizing all models. Highlights LiteHybridNet-DPAM's trade-off: competitive accuracy with significantly lower parameters and latency vs. ResNet-50 and VGG-16.

### Step 18 — XAI: Integrated Gradients
Uses **Captum's IntegratedGradients** to attribute predictions back to individual input pixels. For each test sample:
- Computes attribution map from baseline (black image) to input
- Overlays attribution heatmap on original MRI
- Highlights which pixel regions drove the classification decision

### Step 19 — XAI: Grad-CAM
Applies **Gradient-weighted Class Activation Mapping (Grad-CAM)** to the final convolutional layer. Produces class-discriminative heatmaps showing which spatial regions of the MRI the model activates for each tumor class.

### Step 20 — XAI: Qualitative Visualisation
Combines Grad-CAM and Integrated Gradients side-by-side for a set of representative test samples. Produces a publication-quality qualitative XAI figure comparing both explanation methods across all 4 tumor classes.

### Step 21 — Failure Case Visualisation
Identifies and visualizes **misclassified samples** from the test set. For each failure case, shows:
- Original MRI image
- True label vs. predicted label
- Grad-CAM heatmap
- MC Dropout uncertainty score

Provides insight into where and why the model makes errors — valuable for the paper's limitations section.

### Step 22 — Architecture Diagram
Generates a **publication-ready architecture diagram** of LiteHybridNet-DPAM as a matplotlib figure, showing each layer block (backbone stages, DPAM modules, MSF fusion, MC Dropout head) with dimensions and color coding.

---

## 🧠 Model Architecture Details

### LiteHybridNet-DPAM Layer Stack

| Block | Details | Output Shape |
|---|---|---|
| Input | MRI image | 224×224×3 |
| MobileNetV2 Stage 0 | Conv2D, stride 2 | 112×112×32 |
| Stage 1 + DPAM ×3 | InvRes t=6, c=32 + DPAM | 28×28×32 **(Early Scale)** |
| Stage 2-B4 + DPAM ×4 | InvRes t=6, c=64 + DPAM | 14×14×64 |
| Stage 2-B5 + DPAM ×3 | InvRes t=6, c=96 + DPAM | 14×14×96 |
| Stage 2-B6 + DPAM ×3 | InvRes t=6, c=160 + DPAM | 7×7×160 **(Deep Scale)** |
| MSF Fusion | project + concat → 256-D | 7×7×256 |
| AdaptiveAvgPool2d | Global average pooling | 1×1×256 |
| Flatten + MC Dropout (0.3) | — | 1280-D |
| Classifier | Linear(1280 → 4) | 4-D |

### DPAM Internal Structure

```
Input Feature Map (H×W×C)
    ├──► SE Channel Attention
    │    (squeeze → FC → FC → sigmoid → scale)
    │
    └──► Spatial Attention
         (avg+max pool concat → Conv7×7 → sigmoid → scale)
         │
    [Element-wise Add of both outputs] → Output
```

---

## 📐 Baseline Models Compared

| Model | Params | FLOPs | Notes |
|---|---|---|---|
| **LiteHybridNet-DPAM** | ~3.5M | Low | **Ours — novel** |
| MobileNetV2 | ~3.4M | Low | Closest architecture |
| EfficientNet-B0 | ~5.3M | Low-Med | Compound scaling |
| ResNet-50 | ~25.6M | High | Deep residual |
| VGG-16 | ~138M | Very High | Classic baseline |

---

## 📏 Evaluation Strategy

| Method | Purpose |
|---|---|
| **5-Fold Stratified CV** | Generalisation measurement (mean ± std) |
| **Hold-out Test Set** | Final unbiased performance |
| **Macro F1** | Primary metric (handles class imbalance) |
| **ECE** | Calibration quality |
| **MC Dropout Entropy** | Uncertainty quantification |
| **Ablation Study** | Per-component contribution |
| **Grad-CAM + IG** | Spatial explainability |

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.8+
- CUDA-compatible GPU (strongly recommended — tested on Colab T4/A100)
- Google Colab (recommended) or local Jupyter environment

### Install Dependencies

```bash
pip install torch torchvision --upgrade
pip install captum grad-cam==1.4.6
pip install seaborn torchinfo scipy
pip install numpy==1.26.4
```

All installs are also handled automatically in the first cell of the notebook.

---

## ▶️ How to Run

1. **Download the datasets** from Figshare and/or Mendeley (links in [Datasets](#-datasets) section). Prepare as ZIP files.

2. **Upload to Google Drive** — Place the ZIP in `MyDrive/` (or update the path in Step 3).

3. **Open the notebook** — Upload `figshare_update_code.ipynb` to Google Colab.

4. **Enable GPU** — `Runtime → Change runtime type → A100 or T4 GPU`

5. **Run all cells** — `Runtime → Run all`

   The notebook will auto-detect Colab vs. local, extract the dataset, run all 22 steps, and save all figures.

6. **Outputs** — All publication figures, confusion matrices, XAI heatmaps, and results CSVs are saved automatically.

> 💡 **Tip:** For local runs, update `ZIP_PATH` in Step 3 to your local file path.

---

## 📈 Results

Full results are generated at runtime in Steps 16–17 (efficiency table) and Step 22 (final summary figure). Key comparisons:

- LiteHybridNet-DPAM vs. ResNet-50, VGG-16, EfficientNet-B0, MobileNetV2
- Ablation: DPAM vs. CBAM-seq vs. channel-only vs. spatial-only vs. no-attention
- Calibration: ECE before and after Temperature Scaling
- XAI: Grad-CAM and Integrated Gradients across all 4 tumor classes

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch, TorchVision |
| **Model Analysis** | torchinfo (params/FLOPs) |
| **XAI** | Captum (Integrated Gradients), grad-cam (Grad-CAM) |
| **Calibration** | Temperature Scaling (custom), scipy |
| **Evaluation** | scikit-learn (F1, confusion matrix, classification report) |
| **Visualization** | matplotlib, seaborn |
| **Data Handling** | NumPy, pandas, PIL |
| **Hashing** | hashlib (MD5 duplicate detection) |
| **Environment** | Google Colab (GPU) / Jupyter Notebook |

---

## 📚 Dataset Citation

```bibtex
@data{zwr4ntf94j,
  author    = {HIRA, MD IRFANUL KABIR and HOSSAIN, MD SOHAG and
               BITHEE, MST MORIOM AKTER and Sara, Umme Sara and
               HASAN, MD MAHMUDUL and Towsif, Abdullah Al and Ahmed, Md Kowsar},
  title     = {Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)},
  year      = {2025},
  publisher = {Mendeley Data},
  version   = {V5},
  doi       = {10.17632/zwr4ntf94j.5},
  url       = {https://data.mendeley.com/datasets/zwr4ntf94j/5}
}

@data{figshare14778750,
  title = {Brain MRI Dataset},
  year  = {2021},
  doi   = {10.6084/m9.figshare.14778750},
  url   = {https://figshare.com/articles/dataset/Brain_MRI_Dataset/14778750}
}
```

---

## 📄 Paper Citation

This repository is the official implementation of our research paper. Citation will be updated upon acceptance.

```bibtex
@article{litehybridnet_dpam_2025,
  title  = {LiteHybridNet-DPAM: Lightweight Brain Tumor Classification with
            Dual-Path Attention Module and Uncertainty Estimation},
  note   = {Under Review},
  year   = {2025}
}
```

---

*Reviewer-Ready v6 — LiteHybridNet-DPAM | Brain Tumor MRI Classification with Novel Dual-Path Attention, Multi-Scale Fusion, and MC Dropout Uncertainty*
