import os
import numpy as np

"""
Defining common constant variables for training pipeline
"""

# Generic pipeline settings
PIPELINE_NAME: str = "CropAndFertilizer"
ARTIFACT_DIR: str = "artifacts"

# Data file names
CROP_FILE_NAME: str = "Crop_recommendation.csv"
FERTILIZER_FILE_NAME: str = "Fertilizer_Prediction.csv"

# Crop dataset splits
CROP_TRAIN_FILE_NAME: str = "crop_train.csv"
CROP_TEST_FILE_NAME: str = "crop_test.csv"
CROP_TARGET_COLUMN = "label"

# Fertilizer dataset splits
FERTILIZER_TRAIN_FILE_NAME: str = "fertilizer_train.csv"
FERTILIZER_TEST_FILE_NAME: str = "fertilizer_test.csv"
FERTILIZER_TARGET_COLUMN = "Fertilizer_name"


# Schema files
CROP_SCHEMA_FILE_PATH = os.path.join("data_schema", "crop_schema.yaml")
FERTILIZER_SCHEMA_FILE_PATH = os.path.join("data_schema", "fertilizer_schema.yaml")

# Preprocessing + Model
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
CROP_MODEL_FILE_NAME = "crop_model.pkl"
FERTILIZER_MODEL_FILE_NAME = "fertilizer_model.pkl"
SAVED_MODEL_DIR = os.path.join("saved_models")


"""
Data Ingestion related constants
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# MongoDB
DATA_INGESTION_DATABASE_NAME: str = "AgriDB"
DATA_INGESTION_CROP_COLLECTION_NAME: str = "CropData"
DATA_INGESTION_FERT_COLLECTION_NAME: str = "FertilizerData"
DATA_INGESTION_FARMER_COLLECTION_NAME: str = "FarmerInputs"
ENABLE_MONGO_LOGGING: bool = True

"""
Data Validation related constants
"""

# Root validation directory
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Subfolders for valid/invalid data after validation
DATA_VALIDATION_CROP_VALID_DIR: str = "crop_validated"
DATA_VALIDATION_CROP_INVALID_DIR: str = "crop_invalid"

DATA_VALIDATION_FERTILIZER_VALID_DIR: str = "fertilizer_validated"
DATA_VALIDATION_FERTILIZER_INVALID_DIR: str = "fertilizer_invalid"

# Drift report (to compare current vs reference data)
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

# File name of the drift report
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yml"

"""
Data Transformation related constants
"""

# Root transformation directory
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# Subfolders for transformed data
DATA_TRANSFORMATION_CROP_TRANSFORMED_DATA_DIR: str = "crop_transformed"
DATA_TRANSFORMATION_FERTILIZER_TRANSFORMED_DATA_DIR: str = "fertilizer_transformed"

# Subfolders for serialized preprocessing objects
DATA_TRANSFORMATION_CROP_OBJECT_DIR: str = "crop_transformed_object"
DATA_TRANSFORMATION_FERTILIZER_OBJECT_DIR: str = "fertilizer_transformed_object"

# Imputer parameters (common to both datasets)
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"
}

# Preprocessor object file names
DATA_TRANSFORMATION_CROP_PREPROCESSOR_FILE: str = "crop_preprocessor.pkl"
DATA_TRANSFORMATION_FERTILIZER_PREPROCESSOR_FILE: str = "fertilizer_preprocessor.pkl"

# Transformed dataset file names
DATA_TRANSFORMATION_CROP_TRAIN_FILE: str = "crop_train.npy"
DATA_TRANSFORMATION_CROP_TEST_FILE: str = "crop_test.npy"

DATA_TRANSFORMATION_FERTILIZER_TRAIN_FILE: str = "fertilizer_train.npy"
DATA_TRANSFORMATION_FERTILIZER_TEST_FILE: str = "fertilizer_test.npy"
"""
Model Trainer related constants
"""

# Root trainer directory
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Where trained models will be stored
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"


# Separate model file names
MODEL_TRAINER_CROP_MODEL_NAME: str = "crop_model.pkl"
MODEL_TRAINER_FERTILIZER_MODEL_NAME: str = "fertilizer_model.pkl"

# Expected minimum accuracy / F1-score for acceptance
MODEL_TRAINER_EXPECTED_SCORE: float = 0.8  # Kaggle datasets are relatively clean

# Threshold to detect overfitting/underfitting
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05


# # Optional: cloud storage bucket if using deployment
# TRAINING_BUCKET_NAME = "cropfertilizer-models"
