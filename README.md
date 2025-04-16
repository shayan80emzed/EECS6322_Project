# 🧠 EECS6322 Project: Boosting the Visual Interpretability of CLIP via Adversarial Fine-Tuning

> Reproducibility Study · York University  
> Course: EECS6322 - Neural Networks and Deep Learning  
> Author: Shayan Mohammadizadehsamakosh

---

## 📌 Overview

This project reproduces the results from the paper  
**"Boosting the Visual Interpretability of CLIP via Adversarial Fine-Tuning"**.

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
