from datetime import datetime
import os
from main_files.constants import train_pipeline

print(train_pipeline.PIPELINE_NAME)
print(train_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=train_pipeline.PIPELINE_NAME
        self.artifact_name=train_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp

import os
from main_files.constants import train_pipeline
from main_files.entity.config_entity import TrainingPipelineConfig


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # === Base directory for artifacts ===
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, train_pipeline.DATA_INGESTION_DIR_NAME
        )

        # === Raw CSV paths (static, always inside data/ folder, not artifacts) ===
        self.crop_raw_file_path: str = os.path.join(
            "data", train_pipeline.CROP_FILE_NAME
        )
        self.fertilizer_raw_file_path: str = os.path.join(
            "data", train_pipeline.FERTILIZER_FILE_NAME
        )

        # === Feature store (artifacts) ===
        self.crop_feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            train_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            train_pipeline.CROP_FILE_NAME
        )
        self.fertilizer_feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            train_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            train_pipeline.FERTILIZER_FILE_NAME
        )

        # === Ingested train/test (artifacts) ===
        self.crop_training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            train_pipeline.DATA_INGESTION_INGESTED_DIR,
            train_pipeline.CROP_TRAIN_FILE_NAME
        )
        self.crop_testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            train_pipeline.DATA_INGESTION_INGESTED_DIR,
            train_pipeline.CROP_TEST_FILE_NAME
        )

        self.fertilizer_training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            train_pipeline.DATA_INGESTION_INGESTED_DIR,
            train_pipeline.FERTILIZER_TRAIN_FILE_NAME
        )
        self.fertilizer_testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            train_pipeline.DATA_INGESTION_INGESTED_DIR,
            train_pipeline.FERTILIZER_TEST_FILE_NAME
        )

        # === Train/Test split ratio ===
        self.train_test_split_ratio: float = train_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        # === MongoDB config (optional) ===
        self.crop_collection_name: str = train_pipeline.DATA_INGESTION_CROP_COLLECTION_NAME
        self.fertilizer_collection_name: str = train_pipeline.DATA_INGESTION_FERT_COLLECTION_NAME
        self.farmer_collection_name: str = train_pipeline.DATA_INGESTION_FARMER_COLLECTION_NAME
        self.database_name: str = train_pipeline.DATA_INGESTION_DATABASE_NAME

        # === Mongo toggle (default False, can be enabled externally) ===
        self.mongo_enabled: bool = getattr(train_pipeline, "DATA_INGESTION_MONGO_ENABLED", False)


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,train_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_crop_data_dir:str=os.path.join(self.data_validation_dir,train_pipeline.DATA_VALIDATION_CROP_VALID_DIR)
        self.invalid_crop_data_dir:str=os.path.join(self.data_validation_dir,train_pipeline.DATA_VALIDATION_CROP_INVALID_DIR)
        self.valid_fertilizer_data_dir:str=os.path.join(self.data_validation_dir,train_pipeline.DATA_VALIDATION_FERTILIZER_VALID_DIR)
        self.invalid_fertilizer_data_dir:str=os.path.join(self.data_validation_dir,train_pipeline.DATA_VALIDATION_FERTILIZER_INVALID_DIR)
        self.valid_crop_train_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.CROP_TRAIN_FILE_NAME)
        self.invalid_crop_train_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.CROP_TRAIN_FILE_NAME)
        self.valid_crop_test_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.CROP_TEST_FILE_NAME)    
        self.invalid_crop_test_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.CROP_TEST_FILE_NAME)
        self.invalid_fertilizer_test_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.FERTILIZER_TEST_FILE_NAME)
        self.valid_fertilizer_test_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.FERTILIZER_TEST_FILE_NAME)
        self.valid_fertilizer_train_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.FERTILIZER_TRAIN_FILE_NAME)
        self.invalid_fertilizer_train_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.FERTILIZER_TRAIN_FILE_NAME)
        self.drift_report_file_path:str=os.path.join(self.data_validation_dir,train_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,train_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            train_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # Crop dataset paths
        self.crop_transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, 
            train_pipeline.DATA_TRANSFORMATION_CROP_TRANSFORMED_DATA_DIR, 
            train_pipeline.DATA_TRANSFORMATION_CROP_TRAIN_FILE
        )
        self.crop_transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, 
            train_pipeline.DATA_TRANSFORMATION_CROP_TRANSFORMED_DATA_DIR, 
            train_pipeline.DATA_TRANSFORMATION_CROP_TEST_FILE
        )
        self.crop_preprocessor_object_path: str = os.path.join(
            self.data_transformation_dir, 
            train_pipeline.DATA_TRANSFORMATION_CROP_OBJECT_DIR, 
            train_pipeline.DATA_TRANSFORMATION_CROP_PREPROCESSOR_FILE
        )

        # Fertilizer dataset paths
        self.fertilizer_transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, 
            train_pipeline.DATA_TRANSFORMATION_FERTILIZER_TRANSFORMED_DATA_DIR, 
            train_pipeline.DATA_TRANSFORMATION_FERTILIZER_TRAIN_FILE
        )
        self.fertilizer_transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, 
            train_pipeline.DATA_TRANSFORMATION_FERTILIZER_TRANSFORMED_DATA_DIR, 
            train_pipeline.DATA_TRANSFORMATION_FERTILIZER_TEST_FILE
        )
        self.fertilizer_preprocessor_object_path: str = os.path.join(
            self.data_transformation_dir, 
            train_pipeline.DATA_TRANSFORMATION_FERTILIZER_OBJECT_DIR, 
            train_pipeline.DATA_TRANSFORMATION_FERTILIZER_PREPROCESSOR_FILE
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            train_pipeline.MODEL_TRAINER_DIR_NAME
        )

        # Crop model path
        self.crop_model_file_path: str = os.path.join(
            self.model_trainer_dir, 
            train_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            train_pipeline.MODEL_TRAINER_CROP_MODEL_NAME
        )

        # Fertilizer model path
        self.fertilizer_model_file_path: str = os.path.join(
            self.model_trainer_dir, 
            train_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            train_pipeline.MODEL_TRAINER_FERTILIZER_MODEL_NAME
        )
        

        # Performance thresholds
        self.expected_accuracy: float = train_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = (
            train_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        )
