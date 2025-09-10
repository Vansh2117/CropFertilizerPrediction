import os, sys
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from main_files.constants import train_pipeline
from main_files.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from main_files.entity.config_entity import DataTransformationConfig
from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging
from main_files.utils.main_utils.utils import save_numpy_array_data, save_object



class DataTransformation:
    def __init__(self, 
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CropFertilizerException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def get_data_transformer_object(self, df: pd.DataFrame, target_col: str) -> Pipeline:
        """
        Dynamically builds preprocessing pipeline:
        - Numeric: KNN Imputer + Standard Scaler
        - Categorical: Mode Imputer + OneHotEncoder
        """
        logging.info("Entered get_data_transformer_object method of Transformation class")
        try:
            features = df.drop(columns=[target_col], axis=1)

            num_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_features = features.select_dtypes(include=["object"]).columns.tolist()

            logging.info(f"Numeric features: {num_features}")
            logging.info(f"Categorical features: {cat_features}")

            num_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=3)),
                ("scaler", StandardScaler())
            ])

            transformers = []
            if num_features:
                transformers.append(("num", num_pipeline, num_features))
            if cat_features:
                cat_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ])
                transformers.append(("cat", cat_pipeline, cat_features))

            processor = ColumnTransformer(transformers=transformers)
            return processor

        except Exception as e:
            raise CropFertilizerException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method in DataTransformation class")
        try:
            logging.info("Starting data transformation")

            # ---- Crop Dataset ----
            crop_train_df = self.read_data(self.data_validation_artifact.crop_valid_train_file_path)
            crop_test_df = self.read_data(self.data_validation_artifact.crop_valid_test_file_path)

            crop_target = train_pipeline.CROP_TARGET_COLUMN
            crop_X_train = crop_train_df.drop(columns=[crop_target], axis=1)
            crop_y_train = crop_train_df[crop_target]

            crop_X_test = crop_test_df.drop(columns=[crop_target], axis=1)
            crop_y_test = crop_test_df[crop_target]

            crop_preprocessor = self.get_data_transformer_object(crop_train_df, crop_target)
            crop_preprocessor_object = crop_preprocessor.fit(crop_X_train)

            crop_X_train_transformed = crop_preprocessor_object.transform(crop_X_train)
            crop_X_test_transformed = crop_preprocessor_object.transform(crop_X_test)

            crop_label_encoder = LabelEncoder()
            crop_y_train_encoded = crop_label_encoder.fit_transform(crop_y_train)
            crop_y_test_encoded = crop_label_encoder.transform(crop_y_test)

            crop_train_arr = np.c_[crop_X_train_transformed, crop_y_train_encoded]
            crop_test_arr = np.c_[crop_X_test_transformed, crop_y_test_encoded]

            # ---- Fertilizer Dataset ----
            fert_train_df = self.read_data(self.data_validation_artifact.fertilizer_valid_train_file_path)
            fert_test_df = self.read_data(self.data_validation_artifact.fertilizer_valid_test_file_path)

            fert_target = train_pipeline.FERTILIZER_TARGET_COLUMN
            fert_X_train = fert_train_df.drop(columns=[fert_target], axis=1)
            fert_y_train = fert_train_df[fert_target]

            fert_X_test = fert_test_df.drop(columns=[fert_target], axis=1)
            fert_y_test = fert_test_df[fert_target]

            fert_preprocessor = self.get_data_transformer_object(fert_train_df, fert_target)
            fert_preprocessor_object = fert_preprocessor.fit(fert_X_train)

            fert_X_train_transformed = fert_preprocessor_object.transform(fert_X_train)
            fert_X_test_transformed = fert_preprocessor_object.transform(fert_X_test)

            fert_label_encoder = LabelEncoder()
            fert_y_train_encoded = fert_label_encoder.fit_transform(fert_y_train)
            fert_y_test_encoded = fert_label_encoder.transform(fert_y_test)

            fert_train_arr = np.c_[fert_X_train_transformed, fert_y_train_encoded]
            fert_test_arr = np.c_[fert_X_test_transformed, fert_y_test_encoded]

            # ---- Save numpy arrays + preprocessors ----
            # Crop
            save_numpy_array_data(self.data_transformation_config.crop_transformed_train_file_path, crop_train_arr)
            save_numpy_array_data(self.data_transformation_config.crop_transformed_test_file_path, crop_test_arr)
            save_object(self.data_transformation_config.crop_preprocessor_object_path, crop_preprocessor_object)
            save_object("final_model/crop_label_encoder.pkl", crop_label_encoder)

            # Fertilizer
            save_numpy_array_data(self.data_transformation_config.fertilizer_transformed_train_file_path, fert_train_arr)
            save_numpy_array_data(self.data_transformation_config.fertilizer_transformed_test_file_path, fert_test_arr)
            save_object(self.data_transformation_config.fertilizer_preprocessor_object_path, fert_preprocessor_object)
            save_object("final_model/fertilizer_label_encoder.pkl", fert_label_encoder)


            # ---- Preparing artifacts ----
            data_transformation_artifact = DataTransformationArtifact(
                crop_preprocessor_object_path=self.data_transformation_config.crop_preprocessor_object_path,
                crop_transformed_train_file_path=self.data_transformation_config.crop_transformed_train_file_path,
                crop_transformed_test_file_path=self.data_transformation_config.crop_transformed_test_file_path,
                fertilizer_preprocessor_object_path=self.data_transformation_config.fertilizer_preprocessor_object_path,
                fertilizer_transformed_train_file_path=self.data_transformation_config.fertilizer_transformed_train_file_path,
                fertilizer_transformed_test_file_path=self.data_transformation_config.fertilizer_transformed_test_file_path,
            )

            return data_transformation_artifact

        except Exception as e:
            raise CropFertilizerException(e, sys)
