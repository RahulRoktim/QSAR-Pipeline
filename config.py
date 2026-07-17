# ======================================
# AutoQSAR Configuration
# ======================================

# Dataset
TARGET = "BRAF"
ACTIVITY_TYPE = "IC50"
MIN_COMPOUNDS = 1000

# Explicit ChEMBL target id (skips the search). ChEMBL has several BRAF target
# entries; CHEMBL5145 is the canonical human BRAF that holds the bulk of the
# data - the minor duplicate CHEMBL2331061 has only ~77 activities. Set to None
# to auto-pick the single-protein target with the MOST activities for TARGET.
TARGET_CHEMBL_ID = "CHEMBL5145"

# Paths
RAW_DATASET = "data/raw/raw_dataset.csv"
PROCESSED_DATASET = "data/processed/processed_dataset.csv"
DESCRIPTOR_DATASET = "outputs/descriptors.csv"
SELECTED_DATASET = "outputs/selected_features.csv"

BEST_MODEL_PATH = "models/best_model.pkl"
MODEL_COMPARISON = "outputs/model_comparison.csv"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Features
COMPARE_MODELS = True
HYPERPARAMETER_SEARCH = False

# Plots
SAVE_FEATURE_IMPORTANCE = True
SAVE_PREDICTION_PLOT = True
SAVE_LEARNING_CURVE = True
SAVE_RESIDUAL_PLOT = True

# Validation
RUN_Y_RANDOMIZATION = True
Y_RANDOMIZATION_RUNS = 10

# Explainability
RUN_SHAP = False

# Prediction
DECIMAL_PLACES = 4