from main_files.pipeline.prediction_pipeline import PredictPipeline

# Initialize
predictor = PredictPipeline()

# Starter input for Crop Prediction
# Format: [N, P, K, temperature, humidity, ph, rainfall]
crop_input = [90, 42, 43, 20.5, 82.0, 6.5, 200.0]
print("Recommended Crop:", predictor.predict_crop(crop_input))

# Starter input for Fertilizer Prediction
# Format: [temperature, humidity, moisture, soil_type, crop_type, N, P, K]
fertilizer_input = [26.0, 60.0, 40.0, "Sandy", "Wheat", 80, 40, 40]
print("Recommended Fertilizer:", predictor.predict_fertilizer(fertilizer_input))
