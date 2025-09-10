# pipeline/training_pipeline.py

import sys
from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging

# Import configs and artifacts
from main_files.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from main_files.components.data_ingestion import DataIngestion
from main_files.components.data_validation import DataValidation
from main_files.components.data_transformation import DataTransformation
from main_files.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def start_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion...")
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def start_data_validation(self, data_ingestion_artifact):
        try:
            logging.info("Starting Data Validation...")
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def start_data_transformation(self, data_validation_artifact):
        try:
            logging.info("Starting Data Transformation...")
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            logging.info("Starting Model Training for Crop and Fertilizer...")

            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )

            # Train Crop model
            crop_model_artifact = model_trainer.train_task("crop")
            logging.info(f"Crop model training completed: {crop_model_artifact}")

            # Train Fertilizer model
            fertilizer_model_artifact = model_trainer.train_task("fertilizer")
            logging.info(f"Fertilizer model training completed: {fertilizer_model_artifact}")

            # Return both artifacts together (tuple or dict)
            return {
                "crop": crop_model_artifact,
                "fertilizer": fertilizer_model_artifact
            }

        except Exception as e:
            raise CropFertilizerException(e, sys)


    def run_pipeline(self):
        try:
            logging.info("====== Training Pipeline Started ======")
            
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

            # Step 3: Data Transformation
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)

            # Step 4: Model Training (both crop & fertilizer)
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifact)

            logging.info("====== Training Pipeline Finished Successfully ======")

            # Return both crop and fertilizer artifacts together
            return model_trainer_artifacts

        except Exception as e:
            raise CropFertilizerException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
