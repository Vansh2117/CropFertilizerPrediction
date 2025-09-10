# import os
# import sys
# import numpy as np
# import pandas as pd
# from main_files.utils.main_utils.utils import load_object
# from main_files.exception.exception import CropFertilizerException

# class PredictPipeline:
#     def __init__(self):
#         try:
#             # Load the AdvisoryModel wrapper (handles crop & fertilizer)
#             self.model_crop = load_object("artifacts/09_10_2025_00_53_54/model_trainer/trained_model/crop_model.pkl")
#             self.model_fert = load_object("artifacts/09_10_2025_00_53_54/model_trainer/trained_model/fertilizer_model.pkl")
#             self.crop_encoder=load_object("final_model/crop_label_encoder.pkl")
#             self.fert_encoder=load_object("final_model/fertilizer_label_encoder.pkl")
#         except Exception as e:
#             raise CropFertilizerException(e, sys)

#     def predict_crop(self, features: list):
#         """
#         Predict recommended crop.
#         Args:
#             features: list = [N, P, K, temperature, humidity, ph, rainfall]
#         """
#         try:
#             feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
#             df = pd.DataFrame([features], columns=feature_names)
#             prediction = self.model_crop.predict("crop",df)
#             prediction_int = np.round(prediction).astype(int)
#             predicted_crop=self.crop_encoder.inverse_transform(prediction_int)
#             return predicted_crop[0]  # single prediction
#         except Exception as e:
#             raise CropFertilizerException(e, sys)

#     # def predict_fertilizer(self, features: list):
#     #     """
#     #     Predict recommended fertilizer.
#     #     Args:
#     #         features: list = [temperature, humidity, moisture, soil_type, crop_type, N, P, K]
#     #     """
#     #     try:
#     #         feature_names = ['Temperature', 'Crop_Type', 'Nitrogen', 'Phosphorous', 'Potassium', 'Moisture', 'Soil_Type', 'Humidity']
#     #         df = pd.DataFrame([features], columns=feature_names)
#     #         prediction = self.model_fert.predict("fertilizer",df)
#     #         prediction_int = np.round(prediction).astype(int)
#     #         predicted_fert=self.crop_encoder.inverse_transform(prediction_int)
#     #         return predicted_fert[0]  # single prediction
#     #     except Exception as e:
#     #         raise CropFertilizerException(e, sys)

#     def predict_fertilizer(self, input_features: dict):
#         """
#         input_features: dictionary {feature_name: value, ...} 
#         Example: {"Temperature":26, "Humidity":52, "Moisture":38, ...}
#         """
#         try:
#             # Load preprocessor and model
#             fert_preprocessor = self.model_fert.get_preprocessor("fertilizer")
#             model = self.model_fert.get_model("fertilizer")
#             feature_names = ['Temperature', 'Crop_Type', 'Nitrogen', 'Phosphorous', 'Potassium', 'Moisture', 'Soil_Type', 'Humidity']

#             # Convert input dict to DataFrame
#             df = pd.DataFrame([input_features], columns=feature_names)

#             # Transform and predict
#             df_transformed = fert_preprocessor.transform(df)
#             prediction = model.predict(df_transformed)

#             # Decode target
#             predicted_fert = self.fert_encoder.inverse_transform(prediction.astype(int))
#             return predicted_fert[0]

#         except Exception as e:
#             raise CropFertilizerException(e, sys)


import sys
import pandas as pd
from main_files.utils.main_utils.utils import load_object
from main_files.exception.exception import CropFertilizerException

class PredictPipeline:
    """
    Scalable prediction pipeline for Crop & Fertilizer advisory.
    Handles preprocessing, prediction, and decoding of labels.
    """

    def __init__(self):
        try:
            # --- Crop ---
            self.crop_model = load_object(
                "artifacts/09_10_2025_04_41_24/model_trainer/trained_model/crop_model.pkl"
            )
            self.crop_encoder = load_object("final_model/crop_label_encoder.pkl")

            # --- Fertilizer ---
            self.fert_model = load_object(
                "artifacts/09_10_2025_04_41_24/model_trainer/trained_model/fertilizer_model.pkl"
            )
            self.fert_encoder = load_object("final_model/fertilizer_label_encoder.pkl")

        except Exception as e:
            raise CropFertilizerException(e, sys)

    def predict_crop(self, features: list):
        """
        Predict recommended crop based on SHC extracted features.

        Args:
            features: list = [N, P, K, temperature, humidity, ph, rainfall]

        Returns:
            predicted_crop: str
        """
        try:
            feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            df = pd.DataFrame([features], columns=feature_names)

            #predict
            prediction = self.crop_model.predict("crop",df)

            # Decode target
            predicted_crop = self.crop_encoder.inverse_transform([int(prediction[0])])
            return predicted_crop[0]

        except Exception as e:
            raise CropFertilizerException(e, sys)

    def predict_fertilizer(self, input_features: dict):
        """
        Predict recommended fertilizer based on SHC + intended crop.

        Args:
            input_features: dict with keys like
            {
                "Temperature": 26, "Humidity": 52, "Moisture": 38,
                "Soil_Type": "Sandy", "Crop_Type": "Wheat",
                "Nitrogen": 10, "Phosphorous": 5, "Potassium": 8
            }

        Returns:
            predicted_fertilizer: str
        """
        try:
            feature_names = [
                "Temperature", "Humidity", "Moisture", "Soil_Type", "Crop_Type", "Nitrogen", "Potassium", "Phosphorous"]
            df = pd.DataFrame([input_features], columns=feature_names)

            #predict
            prediction = self.fert_model.predict("fertilizer",df)

            # Decode target
            predicted_fertilizer = self.fert_encoder.inverse_transform([int(prediction[0])])
            return predicted_fertilizer[0]

        except Exception as e:
            raise CropFertilizerException(e, sys)


