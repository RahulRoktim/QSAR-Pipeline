# 🧬 AutoQSAR Pipeline

An end-to-end automated QSAR (Quantitative Structure–Activity Relationship) pipeline built with Python, RDKit, and Scikit-Learn for molecular descriptor generation, feature selection, machine learning, model comparison, and biological activity prediction.

The pipeline automates the complete workflow from downloading molecular activity data from ChEMBL to training, evaluating, and selecting the best predictive QSAR model.

---

# 🚀 Features

- Automatic molecular activity download from ChEMBL
- Molecular data preprocessing and cleaning
- RDKit molecular descriptor calculation
- Automatic feature selection
- Multiple machine learning algorithms
- Automatic best model selection
- 5-Fold Cross Validation
- Model comparison
- Automatic model saving
- Predicted vs Actual visualization
- Residual plot generation
- Modular pipeline architecture

---

# 🔬 Pipeline Workflow

```text
ChEMBL Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
RDKit Descriptor Calculation
      │
      ▼
Feature Selection
      │
      ▼
Train Multiple Models
      │
      ▼
Cross Validation
      │
      ▼
Automatic Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Evaluation
      │
      ▼
Prediction
```

---

# 📂 Project Structure

```text
QSAR-Pipeline/

├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── chembl_downloader.py
│   ├── preprocess.py
│   ├── descriptors.py
│   ├── feature_selection.py
│   ├── feature_selector.py
│   ├── trainer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── comparison.py
│   ├── models.py
│   ├── tuning.py
│   ├── visualization.py
│   └── utils.py
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/RahulRoktim/QSAR-Pipeline.git
```

Move into the project

```bash
cd QSAR-Pipeline
```

Create a virtual environment

```bash
python -m venv rdkit_env
```

Activate

### Windows

```bash
rdkit_env\Scripts\activate
```

### Linux / macOS

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

The pipeline automatically performs:

- Download ChEMBL dataset
- Data preprocessing
- Molecular descriptor calculation
- Feature selection
- Multiple model training
- Cross-validation
- Model comparison
- Automatic best model selection
- Model evaluation
- Prediction

---

# 🤖 Machine Learning Models

The pipeline currently compares:

- Random Forest
- Extra Trees
- Gradient Boosting

Additional models such as XGBoost, LightGBM, CatBoost, and Support Vector Regression can be enabled if installed.

---

# 📊 Evaluation Metrics

The models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- 5-Fold Cross Validation

The pipeline automatically selects the model with the highest predictive performance.

---

# 📈 Generated Outputs

Running the pipeline automatically generates:

- Processed dataset
- Molecular descriptors
- Selected features
- Model comparison table
- Best trained model
- Predicted vs Actual plot
- Residual plot

---

# 📚 Technologies Used

- Python
- RDKit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Joblib

---

# 🎯 Future Improvements

Planned future enhancements include:

- SHAP Explainability
- Applicability Domain Analysis
- Learning Curves
- Hyperparameter Optimization
- External Validation
- PDF Report Generation
- Batch Prediction
- Deep Learning QSAR Models

---

# 🧪 Research Applications

This pipeline can be adapted for:

- Drug Discovery
- Lead Optimization
- Virtual Screening
- Bioactivity Prediction
- QSAR Modeling
- Computer-Aided Drug Design (CADD)
- AI-Assisted Drug Discovery
- Computational Medicinal Chemistry

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
- ChEMBL Database
- Scikit-Learn Developers
- Scientific Python Community

---

# 📜 License

This project is released under the MIT License.
