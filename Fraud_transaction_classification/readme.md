# Fraud Transaction Classification (Credit Card Fraud Detection)

A production-style machine learning pipeline to classify credit card transactions as **Fraud (1)** or **Legit (0)** using the Kaggle `creditcard.csv` dataset.  
This project focuses on **imbalanced classification**, prioritizing **PR-AUC**, **Recall**, and **threshold optimization** (not just accuracy).

---

## Dataset
- File: `creditcard.csv`
- Rows: 284,807
- Columns: 31
- Features: `Time`, `Amount`, `V1..V28` (PCA-transformed)
- Label: `Class` (0 = Legit, 1 = Fraud)
- Fraud rate: ~0.17% (highly imbalanced)

---

## Project Highlights (Resume-Ready)
- Built an end-to-end ML pipeline with **data validation**, **preprocessing**, **model training**, and **evaluation**.
- Addressed severe class imbalance with **SMOTE inside a Pipeline** (avoids leakage).
- Used **Stratified K-Fold CV** and selected models based on **PR-AUC (Average Precision)**.
- Performed **threshold tuning** to optimize fraud detection performance.

---

## Repository Structure
