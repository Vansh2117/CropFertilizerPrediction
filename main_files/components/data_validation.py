import os
import sys
import pandas as pd
import numpy as np
from typing import Dict
from scipy.stats import ks_2samp

from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging
from main_files.entity.config_entity import DataValidationConfig
from main_files.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from main_files.constants import train_pipeline
from main_files.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # schema files for crop and fertilizer
            self._crop_schema = read_yaml_file(train_pipeline.CROP_SCHEMA_FILE_PATH)
            self._fertilizer_schema = read_yaml_file(train_pipeline.FERTILIZER_SCHEMA_FILE_PATH)

        except Exception as e:
            raise CropFertilizerException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame, schema: Dict) -> bool:
        try:
            required_columns = schema["columns"]
            logging.info(f"Expected columns: {required_columns}")
            logging.info(f"DataFrame columns: {list(dataframe.columns)}")

            return set(required_columns) == set(dataframe.columns)
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def validate_numerical_columns(self, dataframe: pd.DataFrame, schema: dict) -> bool:
        try:
            expected_numerical = set(schema["numerical_columns"])
            actual_cols = set(dataframe.columns)

            missing = expected_numerical - actual_cols
            if missing:
                logging.error(f"Missing numerical columns: {missing}")
                return False

            for col in expected_numerical:
                if not pd.api.types.is_numeric_dtype(dataframe[col]):
                    logging.error(f"Column {col} is not numeric. Found dtype: {dataframe[col].dtype}")
                    return False

            return True
        except Exception as e:
            raise CropFertilizerException(e, sys)
    
    def validate_categorical_columns(self, dataframe: pd.DataFrame, schema: dict) -> bool:
        try:
            expected_categorical_cols = set(schema.get("categorical_columns", []))
            for col in expected_categorical_cols:
                if col not in dataframe.columns:
                    logging.error(f"Categorical column missing: {col}")
                    return False
                if not pd.api.types.is_object_dtype(dataframe[col]) and not pd.api.types.is_categorical_dtype(dataframe[col]):
                    logging.error(f"Column {col} is not categorical. Found {dataframe[col].dtype}")
                    return False
            return True
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame,
                          threshold: float = 0.05, dataset_name: str = "dataset") -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1, d2 = base_df[column], current_df[column]
                test_result = ks_2samp(d1, d2)
                p_value=test_result.pvalue
                drift_found = p_value < threshold

                if drift_found:
                    status = False

                report[column] = {
                    "p_value": float(p_value),
                    "drift_status": drift_found
                }

            # Write report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(drift_report_file_path, {dataset_name: report}, replace=False)

            return status
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def _validate_dataset(self, train_file, test_file, schema, dataset_name):
        """Validate one dataset (crop or fertilizer)."""
        try:
            train_df = self.read_data(train_file)
            test_df = self.read_data(test_file)

            expected_columns = schema["columns"]

            # column count validation
            if not self.validate_number_of_columns(train_df, schema):
                extra = set(train_df.columns) - set(expected_columns)
                missing = set(expected_columns) - set(train_df.columns)
                raise CropFertilizerException(
                    f"{dataset_name} train file has invalid columns.\n"
                    f"Expected: {expected_columns}\n"
                    f"Found: {list(train_df.columns)}\n"
                    f"Missing: {list(missing)}\n"
                    f"Extra: {list(extra)}",
                    sys
                )

            if not self.validate_number_of_columns(test_df, schema):
                extra = set(test_df.columns) - set(expected_columns)
                missing = set(expected_columns) - set(test_df.columns)
                raise CropFertilizerException(
                    f"{dataset_name} test file has invalid columns.\n"
                    f"Expected: {expected_columns}\n"
                    f"Found: {list(test_df.columns)}\n"
                    f"Missing: {list(missing)}\n"
                    f"Extra: {list(extra)}",
                    sys
                )

            # numeric column validation
            if not self.validate_numerical_columns(train_df, schema):
                raise CropFertilizerException(f"{dataset_name} train file has non-numeric issues", sys)
            if not self.validate_numerical_columns(test_df, schema):
                raise CropFertilizerException(f"{dataset_name} test file has non-numeric issues", sys)

            # categorical column validation (if schema defines any)
            if not self.validate_categorical_columns(train_df, schema):
                raise CropFertilizerException(f"{dataset_name} train file has categorical column issues", sys)
            if not self.validate_categorical_columns(test_df, schema):
                raise CropFertilizerException(f"{dataset_name} test file has categorical column issues", sys)

            # drift detection
            drift_status = self.detect_data_drift(train_df, test_df, dataset_name=dataset_name)

            return train_df, test_df, drift_status

        except Exception as e:
            raise CropFertilizerException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # --- Crop dataset ---
            crop_train_df, crop_test_df, crop_drift = self._validate_dataset(
                self.data_ingestion_artifact.crop_train_file_path,
                self.data_ingestion_artifact.crop_test_file_path,
                self._crop_schema,
                "crop"
            )

            # --- Fertilizer dataset ---
            fert_train_df, fert_test_df, fert_drift = self._validate_dataset(
                self.data_ingestion_artifact.fertilizer_train_file_path,
                self.data_ingestion_artifact.fertilizer_test_file_path,
                self._fertilizer_schema,
                "fertilizer"
            )

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_crop_train_file_path), exist_ok=True)

            crop_train_df.to_csv(self.data_validation_config.valid_crop_train_file_path, index=False)
            crop_test_df.to_csv(self.data_validation_config.valid_crop_test_file_path, index=False)
            fert_train_df.to_csv(self.data_validation_config.valid_fertilizer_train_file_path, index=False)
            fert_test_df.to_csv(self.data_validation_config.valid_fertilizer_test_file_path, index=False)

            validation_status = crop_drift and fert_drift

            return DataValidationArtifact(
                validation_status=validation_status,
                crop_valid_train_file_path=self.data_validation_config.valid_crop_train_file_path,
                crop_valid_test_file_path=self.data_validation_config.valid_crop_test_file_path,
                crop_invalid_train_file_path=None,
                crop_invalid_test_file_path=None,
                fertilizer_valid_train_file_path=self.data_validation_config.valid_fertilizer_train_file_path,
                fertilizer_valid_test_file_path=self.data_validation_config.valid_fertilizer_test_file_path,
                fertilizer_invalid_train_file_path=None,
                fertilizer_invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise CropFertilizerException(e, sys)
