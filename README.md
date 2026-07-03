# 🧬 AutoQSAR Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end QSAR (Quantitative Structure–Activity Relationship) pipeline for molecular descriptor generation, feature selection, machine learning, and biological activity prediction using **RDKit** and **Scikit-Learn**.

This project automates the complete workflow from downloading molecular data from **ChEMBL** to training and validating predictive QSAR models.

---

# 📌 Features

- Download molecular activity data from ChEMBL
- Molecular data preprocessing and cleaning
- RDKit molecular descriptor calculation
- Automatic feature selection
- Random Forest regression model
- 5-Fold Cross Validation
- Y-Randomization (Response Permutation Test)
- Feature importance analysis
- Predicted vs Actual visualization
- Automatic model saving

---

# 🔬 Pipeline Workflow

```
ChEMBL Dataset
       │
       ▼
Preprocessing
       │
       ▼
RDKit Descriptor Calculation
       │
       ▼
Feature Selection
       │
       ▼
Random Forest Training
       │
       ▼
5-Fold Cross Validation
       │
       ▼
Y-Randomization Test
       │
       ▼
Model Evaluation
       │
       ▼
Prediction
```

---

# 📂 Project Structure

```
QSAR-Pipeline/

├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── chembl_downloader.py
│   ├── preprocess.py
│   ├── descriptors.py
│   ├── feature_selection.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
├── outputs/
│
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/RahulRoktim/QSAR-Pipeline.git

cd QSAR-Pipeline
```

Create a virtual environment

```bash
python -m venv rdkit_env
```

Activate

Windows

```bash
rdkit_env\Scripts\activate
```

Linux / macOS

```bash
source rdkit_env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

Run the complete pipeline

```bash
python main.py
```

The pipeline automatically performs

1. Download ChEMBL dataset
2. Data preprocessing
3. Descriptor calculation
4. Feature selection
5. Model training
6. Cross-validation
7. Y-randomization
8. Model evaluation

---

# 📊 Example Performance

Current implementation achieved:

| Metric | Value |
|--------|-------:|
| Test R² | 0.628 |
| MAE | 0.605 |
| RMSE | 0.783 |
| 5-Fold CV R² | 0.685 ± 0.047 |
| Average Random R² | -0.161 |

The negative R² values obtained during Y-Randomization indicate that the trained model captures genuine structure–activity relationships rather than learning random correlations.

---

# 📈 Generated Outputs

The pipeline automatically generates

- Processed dataset
- Molecular descriptors
- Selected features
- Feature importance CSV
- Feature importance plot
- Predicted vs Actual plot
- Trained Random Forest model

---

# 🧪 Machine Learning Model

Current model:

- Random Forest Regressor

Validation methods:

- Train/Test Split
- 5-Fold Cross Validation
- Y-Randomization Test

Evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# 📚 Technologies Used

- Python
- RDKit
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Joblib

---

# 🎯 Future Improvements

Planned enhancements include:

- Hyperparameter optimization
- Scikit-Learn Pipeline integration
- Additional ML algorithms
  - XGBoost
  - LightGBM
  - Support Vector Regression
  - CatBoost
- SHAP explainability
- Applicability Domain analysis
- External validation datasets
- Deep Learning QSAR models

---

# 📖 Research Applications

This project can be adapted for:

- Drug Discovery
- Lead Optimization
- Virtual Screening
- Bioactivity Prediction
- QSAR Modeling
- Computational Medicinal Chemistry
- Computer-Aided Drug Design (CADD)

---

# 👨‍💻 Author

**Rahul Roktim**

Bachelor of Pharmacy  
Daffodil International University  
Bangladesh

GitHub:

https://github.com/RahulRoktim

---

# ⭐ Acknowledgements

- RDKit Development Team
- Scikit-Learn Developers
- ChEMBL Database
- Open Source Scientific Python Community
