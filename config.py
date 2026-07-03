# ==========================
# AutoQSAR Configuration
# ==========================

# Target protein name
TARGET = "EGFR"

# Activity type
ACTIVITY_TYPE = "IC50"

# Minimum number of compounds to retrieve
MIN_COMPOUNDS = 1000

# Output file
RAW_DATASET = "data/raw/raw_dataset.csv"
PROCESSED_DATASET = "data/processed/processed_dataset.csv"
DESCRIPTOR_DATASET = "outputs/descriptors.csv"
MODEL_PATH = "models/random_forest.pkl"