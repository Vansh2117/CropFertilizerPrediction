# main.py
import sys
from main_files.pipeline.training_pipeline import TrainingPipeline
from main_files.entity.config_entity import TrainingPipelineConfig
from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging

if __name__ == "__main__":
    try:
        logging.info("=== Starting Full Training Pipeline ===")
        
        # Initialize pipeline config
        training_pipeline_config = TrainingPipelineConfig()

        # Initialize pipeline
        pipeline = TrainingPipeline()

        # Run pipeline
        model_trainer_artifacts = pipeline.run_pipeline()

        logging.info(f"Training pipeline completed successfully.")
        logging.info(f"Model Trainer Artifacts: {model_trainer_artifacts}")

    except Exception as e:
        raise CropFertilizerException(e, sys)
