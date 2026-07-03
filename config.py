# ======================================
# AutoQSAR Configuration
# ======================================

# Dataset
TARGET = "EGFR"
ACTIVITY_TYPE = "IC50"
MIN_COMPOUNDS = 1000

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