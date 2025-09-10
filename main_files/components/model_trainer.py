import os, sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging
from main_files.entity.artifact_entity import (
    DataTransformationArtifact, 
    ModelTrainerArtifact
)
from main_files.entity.config_entity import ModelTrainerConfig
from main_files.utils.main_utils.utils import (
    load_numpy_array_data, save_object, load_object, evaluate_models
)
from main_files.utils.ml_utils.model.estimator import AdvisoryModel
from main_files.utils.ml_utils.metric.classification_metric import get_classification_score


class ModelTrainer:
    def __init__(
        self, 
        model_trainer_config: ModelTrainerConfig, 
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.advisory_model = AdvisoryModel()  # wrapper for multiple tasks
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def _get_models_and_params(self):
        """Return model candidates and hyperparameters for tuning."""
        models = {
            "Random Forest": RandomForestClassifier(verbose=0, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(verbose=0, random_state=42),
            "Adaboost": AdaBoostClassifier(random_state=42),
        }

        params = {
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "criterion": ["gini", "entropy"]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.1, 0.05, 0.01],
                "subsample": [0.8, 1.0],
                "max_depth": [3, 5]
            },
            "Adaboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.5, 1.0]
            },
        }

        return models, params

    def train_task(self, task_name: str) -> ModelTrainerArtifact:
        """
        Train a specific advisory task (crop or fertilizer).
        Returns: ModelTrainerArtifact
        """
        try:
            logging.info(f"==== Training started for task: {task_name} ====")

            # Load arrays for the chosen task
            if task_name == "crop":
                train_file = self.data_transformation_artifact.crop_transformed_train_file_path
                test_file = self.data_transformation_artifact.crop_transformed_test_file_path
                model_file_path = self.model_trainer_config.crop_model_file_path

            elif task_name == "fertilizer":
                train_file = self.data_transformation_artifact.fertilizer_transformed_train_file_path
                test_file = self.data_transformation_artifact.fertilizer_transformed_test_file_path
                model_file_path = self.model_trainer_config.fertilizer_model_file_path

            else:
                raise ValueError(f"Unsupported task: {task_name}")

            train_arr = load_numpy_array_data(train_file)
            test_arr = load_numpy_array_data(test_file)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, :-1], 
                test_arr[:, -1]
            )

            models, params = self._get_models_and_params()
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # --- Select best model based on primary_metric (f1-score) ---
            # best_model_name = max(model_report, key=lambda name: model_report[name]["primary_metric"])
            # best_model_info = model_report[best_model_name]
            # best_model = best_model_info["best_model"]
            # best_model_name = max(model_report, key=lambda name: model_report[name]["cv_score"])
            # best_model = model_report[best_model_name]["best_model"]
            # best_params = model_report[best_model_name]["best_params"]

            # logging.info(f"Best model for {task_name}: {best_model_name} with params {best_params}")

            # # Train/Test metrics
            # classification_train_metric = best_model["train_metrics"]
            # classification_test_metric = best_model["test_metrics"]
            # Pick best model based on CV score
            best_model_name = max(model_report, key=lambda name: model_report[name]["primary_metric"])
            best_model_info = model_report[best_model_name]

            best_model = best_model_info["best_model"]
            best_params = best_model_info["best_params"]
            classification_train_metric = best_model_info["train_metrics"]
            classification_test_metric = best_model_info["test_metrics"]

            logging.info(
                f"Best model for {task_name}: {best_model_name} "
                f"with params {best_params} "
                f"(CV score = {best_model_info['cv_score']:.4f})"
            )


            # Load preprocessor and wrap in AdvisoryModel
            if task_name == "crop":
                preprocessor = load_object(self.data_transformation_artifact.crop_preprocessor_object_path)
            else:
                preprocessor = load_object(self.data_transformation_artifact.fertilizer_preprocessor_object_path)

            self.advisory_model.register_model(task_name, preprocessor, best_model)

            # Save model wrapper
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            save_object(model_file_path, self.advisory_model)

            # Artifact
            artifact = ModelTrainerArtifact(
                trained_model_file_path=model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"==== Training completed for {task_name} ====")
            return artifact

        except Exception as e:
            raise CropFertilizerException(e, sys)
