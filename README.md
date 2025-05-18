# DoU Dengue Prediction

This repository contains code and resources for predicting dengue outbreaks using the Degree of Urbanization (DoU) as a critical feature. It includes preprocessing scripts, feature extraction modules, and machine learning models aimed at improving the accuracy of dengue prediction by incorporating urbanization metrics from satellite imagery.

## 📁 Repository Structure

```
DoU_dengue_prediction/
├── data/                      # Raw and processed data files
├── vae/                       # Variational Autoencoder scripts -- Model extracted from SatelitteBench
├── DoU_Dengue_pred.ipynb      # Main dengue prediction notebook (with DoU, modified for Municipality 50001 - Villavicencio, Colombia)
├── No_DoU_Dengue_pred.ipynb   # Baseline dengue prediction notebook (without DoU, modified SateliteBench's code for Municipality 50001 - Villavicencio, Colombia)
├── DoU_Classification.py      # Urbanization classification scripts for extracting and analyzing urbanization metrics
└──.gitignore                 # Git ignore file
```

## 🌐 Project Overview

Dengue fever outbreaks are often influenced by urbanization patterns. This project aims to review how dengue prediction is impacted by integrating Degree of Urbanization (DoU) metrics extracted from satellite images, along with conventional environmental and demographic features.

## 🚀 Getting Started

### Prerequisites
Ensure you have the following packages installed:
- Python 3.10+
- numpy
- pandas
- geopandas
- rasterio
- scikit-image
- scikit-learn

### Data Requirements
- Temperature, precipitation, and dengue data is given for Columbia
- Satellite Data is processed into DoU_Classification.py to give DoU labeled data

### Installation
Clone the repository:
```bash
git clone https://github.com/imishaj/DoU_dengue_prediction.git
cd DoU_dengue_prediction
```

## 📊 Usage

### 1. Run the Baseline Model (Without DoU)
```bash
jupyter notebook No_DoU_Dengue_pred.ipynb
```
This notebook runs the baseline dengue prediction model without using DoU features, modified specifically for Municipality 50001 - Villavicencio, Colombia.

### 2. Run the DoU-Enhanced Model
```bash
jupyter notebook DoU_Dengue_pred.ipynb
```
This notebook includes DoU metrics in the dengue prediction pipeline for Municipality 50001 - Villavicencio, Colombia, improving prediction accuracy by incorporating urbanization data.

### 3. Urbanization Classification
```bash
python DoU_Classification.py
```
This script is used to extract and classify urbanization metrics from satellite imagery, providing key inputs for the DoU-enhanced model.

## 📈 Results and Metrics
Evaluate the performance of the DoU-enhanced model within the provided notebooks. The results are typically measured using RMSE, MAE, and F1 scores.

---
