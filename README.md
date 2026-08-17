# 🍃 Plant Pathology 2020 - Apple Leaf Disease Classification

Deep learning pipeline for multi-class apple leaf disease classification (**Healthy, Multiple Diseases, Rust, Scab**) based on the **Kaggle Plant Pathology 2020 - FGVC7** challenge.

---

## 📌 Project Overview
* **Architecture:** Transfer Learning with `tf_efficientnet_b0_ns` (timm)
* **Augmentations:** Albumentations (ShiftScaleRotate, ColorJitter, CoarseDropout, Flips)
* **Imbalance Handling:** Inverse Class Frequency Weighting for CrossEntropyLoss
* **Hardware Acceleration:** Apple Silicon MPS (Metal Performance Shaders) support
* **Evaluation Metric:** **0.9837 Validation Multi-Class ROC-AUC**

---

## 📂 Repository Structure
```text
├── plant_pathology_classification.ipynb  # End-to-end data prep, training & inference
├── requirements.txt                     # Project dependencies
├── sample_submission.csv                # Sample output format
└── README.md                            # Documentation
