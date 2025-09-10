from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging

# Configs and Artifacts
from main_files.entity.config_entity import DataIngestionConfig
from main_files.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

# For optional MongoDB farmer data ingestion
from main_files.etl.mongo_client import MongoDBClient


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Data Ingestion for Crop & Fertilizer datasets.
        Supports hybrid mode: local CSVs + farmer inputs from MongoDB.
        """
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def load_local_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load base datasets from raw CSVs in data/ and copy to feature_store."""
        try:
                # Read raw CSVs
                crop_df = pd.read_csv(self.config.crop_raw_file_path)
                fert_df = pd.read_csv(self.config.fertilizer_raw_file_path)
                logging.info("Loaded crop and fertilizer raw datasets from data/ folder")

                # Ensure feature store dir exists
                os.makedirs(os.path.dirname(self.config.crop_feature_store_file_path), exist_ok=True)

                # Save a copy into feature store
                crop_df.to_csv(self.config.crop_feature_store_file_path, index=False, header=True)
                fert_df.to_csv(self.config.fertilizer_feature_store_file_path, index=False, header=True)
                logging.info("Copied datasets into feature_store inside artifacts")

                return crop_df, fert_df

        except Exception as e:
                raise CropFertilizerException(e, sys)

    def load_farmer_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load farmer inputs from MongoDB collections (if enabled)."""
        try:
            if not getattr(self.config, "mongo_enabled", False):
                logging.info("MongoDB not enabled, skipping farmer data ingestion")
                return pd.DataFrame(), pd.DataFrame()

            mongo_client = MongoDBClient()
            crop_df = mongo_client.find_as_df(self.config.crop_collection_name)
            fert_df = mongo_client.find_as_df(self.config.fertilizer_collection_name)
            mongo_client.close()

            logging.info(f"Loaded farmer data from MongoDB: "
                         f"{crop_df.shape[0]} crop rows, {fert_df.shape[0]} fertilizer rows")

            return crop_df, fert_df
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def merge_datasets(self, base_df: pd.DataFrame, farmer_df: pd.DataFrame) -> pd.DataFrame:
        """Merge farmer data into base dataset if available"""
        if farmer_df is not None and not farmer_df.empty:
            merged = pd.concat([base_df, farmer_df], ignore_index=True)
            logging.info(f"Merged dataset: {base_df.shape[0]} + {farmer_df.shape[0]} rows")
            return merged
        return base_df

    def split_and_save(self, df: pd.DataFrame, train_path: str, test_path: str):
        """Split dataset into train/test and save to CSV"""
        try:
            train_set, test_set = train_test_split(df, test_size=self.config.train_test_split_ratio,random_state=42)

            dir_path = os.path.dirname(train_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            logging.info(f"Saved train/test data: {train_path}, {test_path}")
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # Step 1: Load local data
            crop_base, fert_base = self.load_local_data()

            # Step 2: Load farmer data (if available)
            crop_farmer, fert_farmer = self.load_farmer_data()

            # Step 3: Merge datasets
            crop_final = self.merge_datasets(crop_base, crop_farmer)
            fert_final = self.merge_datasets(fert_base, fert_farmer)

            # Step 4: Split & Save
            self.split_and_save(crop_final, self.config.crop_training_file_path, self.config.crop_testing_file_path)
            self.split_and_save(fert_final, self.config.fertilizer_training_file_path, self.config.fertilizer_testing_file_path)

            # Step 5: Return artifact
            artifact = DataIngestionArtifact(
                crop_train_file_path=self.config.crop_training_file_path,
                crop_test_file_path=self.config.crop_testing_file_path,
                fertilizer_train_file_path=self.config.fertilizer_training_file_path,
                fertilizer_test_file_path=self.config.fertilizer_testing_file_path,
                mongo_enabled=getattr(self.config, "mongo_enabled", False),
                crop_collection_name=self.config.crop_collection_name,
                fertilizer_collection_name=self.config.fertilizer_collection_name,
                farmer_collection_name=self.config.farmer_collection_name,
            )
            logging.info("Data Ingestion completed successfully")
            return artifact

        except Exception as e:
            raise CropFertilizerException(e, sys)
