"""AutoQSAR default configuration.

Every value here is a *default*: `main.py` exposes each one as a command-line
flag and overrides this module before any pipeline stage is imported. Editing
this file changes the defaults for every future run, so prefer flags for
one-off experiments and keep the committed defaults meaningful.
"""

# ======================================
# Dataset
# ======================================

# Human-readable target label. Used to name artefacts AND checked against the
# resolved ChEMBL record before anything is downloaded.
TARGET = "BRAF"

ACTIVITY_TYPE = "IC50"
MIN_COMPOUNDS = 1000

# Curated ChEMBL ids for targets whose entries we have verified by hand.
#
# ChEMBL frequently holds several entries for one protein and the minor
# duplicates carry almost no data - the wrong BRAF entry (CHEMBL2331061) has
# only ~77 activities, which is far too few to train on. Recording the verified
# id per target means the knowledge survives a change of TARGET, instead of
# being stranded in a single global that silently applies to the wrong protein.
KNOWN_TARGET_IDS = {
    "BRAF": "CHEMBL5145",   # Serine/threonine-protein kinase B-raf, Homo sapiens
}

# Explicit override. Leave as None to use KNOWN_TARGET_IDS, falling back to a
# ChEMBL search that picks the single-protein target with the most activities.
# Whatever is used, it is verified against TARGET before download - a mismatch
# raises TargetMismatchError rather than producing a mislabelled dataset.
TARGET_CHEMBL_ID = None

# ======================================
# Paths
# ======================================
RAW_DATASET = "data/raw/raw_dataset.csv"
PROCESSED_DATASET = "data/processed/processed_dataset.csv"
DESCRIPTOR_DATASET = "outputs/descriptors.csv"
SELECTED_DATASET = "outputs/selected_features.csv"

BEST_MODEL_PATH = "models/best_model.pkl"
MODEL_COMPARISON = "outputs/model_comparison.csv"

# ======================================
# Training
# ======================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

COMPARE_MODELS = True
HYPERPARAMETER_SEARCH = False

# ======================================
# Plots
# ======================================
SAVE_FEATURE_IMPORTANCE = True
SAVE_PREDICTION_PLOT = True
SAVE_LEARNING_CURVE = True
SAVE_RESIDUAL_PLOT = True

# ======================================
# Validation
# ======================================
# y-randomization is the control that distinguishes a real structure-activity
# relationship from a model memorising noise. Keep it on for anything reported.
RUN_Y_RANDOMIZATION = True
Y_RANDOMIZATION_RUNS = 10

# ======================================
# Explainability / prediction
# ======================================
RUN_SHAP = False
DECIMAL_PLACES = 4
