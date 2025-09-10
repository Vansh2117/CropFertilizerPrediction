from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    # Local file paths
    crop_train_file_path: str
    crop_test_file_path: str
    fertilizer_train_file_path: str
    fertilizer_test_file_path: str

    # Mongo toggle + collections
    mongo_enabled: bool = False
    crop_collection_name: Optional[str] = None
    fertilizer_collection_name: Optional[str] = None
    farmer_collection_name: Optional[str] = None


@dataclass
class DataValidationArtifact:
    validation_status: bool

    # Crop dataset validation
    crop_valid_train_file_path: str
    crop_valid_test_file_path: str
    crop_invalid_train_file_path: str
    crop_invalid_test_file_path: str

    # Fertilizer dataset validation
    fertilizer_valid_train_file_path: str
    fertilizer_valid_test_file_path: str
    fertilizer_invalid_train_file_path: str
    fertilizer_invalid_test_file_path: str

    # Drift report (single for both datasets)
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    # Crop
    crop_preprocessor_object_path: str
    crop_transformed_train_file_path: str
    crop_transformed_test_file_path: str

    # Fertilizer
    fertilizer_preprocessor_object_path: str
    fertilizer_transformed_train_file_path: str
    fertilizer_transformed_test_file_path: str



@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

