# EECS6322 Project: Boosting the Visual Interpretability of CLIP via Adversarial Fine-Tuning

> Reproducibility Challenge · York University  
> Course: EECS6322 - Neural Networks and Deep Learning  
> Author: Shayan Mohammadizadehsamakosh

---

## 📌 Overview

This project reproduces the results from the paper  
**"Boosting the Visual Interpretability of CLIP via Adversarial Fine-Tuning"** accepted in ICLR 2025. 

We implement and test the proposed **Adversarial Fine-Tuning (AFT)** framework to improve the **visual interpretability** of the CLIP model’s vision encoder, without using any labels or modifying the text encoder.

- Fine-tuning is done adversarially on images only (ImageNet).
- Evaluations include saliency maps, attention maps, and ROAR analysis.
- Results confirm improved interpretability, aligning with the original paper’s claims.

---

## 🗂️ Project Structure

```bash
COMPLETE TREE

.
├── main.py                # Entry point: sets up config, logger, and runs training
├── train.py               # Main training loop and multi-GPU support
├── pgd_huber.py           # PGD implementation with Huber and L-infinity norms
├── config.py              # Config files for training runs (dataclass)
├── outputs/               # Directory to store trained model weights and loggings
│   ├── logs/              # Loggings during the training
│   └── checkpoints/       # Will be created if you try training the model (not included because the .pt files are huge!)
├── results/               # Contains saliency maps, attention visualizations, ROAR outputs
│   ├── SG_GC_ATT.ipynb    # Pipeline for feature attribution by Simple Gradient and GradCAM, plus visualization of attention maps of ViT
│   └── ROAR/              # Pipleline for feature importance analysis
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)


```


---

## 🚀 Methodology

### 🔧 Adversarial Fine-Tuning (AFT)

- Fine-tunes only CLIP's image encoder using PGD-based adversarial training.
- Introduces a norm-regularized objective to smooth image embeddings while preserving similarity to text.
- Uses both **Huber norm** and **L∞ norm** variants.
- Trained on ImageNet for 600 steps using RTX 6000 GPUs.

### 🧪 Interpretability Evaluation

- **Simple Gradient**: Basic gradient attribution
- **Grad-CAM**: Class-specific heatmaps
- **Attention Maps**: Visualization of ViT attention layers
- **ROAR (Remove and Retrain)**: Evaluates how performance drops when important pixels are removed

Interpretability notebooks visualize and compare the original CLIP model with both the reproduced and paper-provided fine-tuned models.

---

## 📊 Results

- **Saliency maps** became significantly more focused and aligned with semantic content.
- **Attention maps** were clearer and more structured post fine-tuning.
- **ROAR analysis** showed a performance drop but was not consistent with the paper in terms of the slope of the drop.

All interpretability evaluation results are available in the `results/` folder. 

My pretrained checkpoints (adversarially fine-tuned models) can be found here: 

[ViT-B-16, L-Infinity Regularizatoin](https://drive.google.com/file/d/1RwSsiPZCJAxUwpyof3_Kqh9weafR9STI/view?usp=sharing) 

[ViT-B-16, Hurber Regularization](https://drive.google.com/file/d/1jya6xPalov3jAS9FafsuA6kKKM5qY4Db/view?usp=sharing)

---

## 🔨 Setup

```bash
git clone https://github.com/shayan80emzed/EECS6322_Project.git
cd EECS6322_Project
pip install -r requirements.txt
python main.py
```
You can change the configs from config.py dataclass.
