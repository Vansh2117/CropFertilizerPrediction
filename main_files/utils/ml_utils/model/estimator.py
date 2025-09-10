import sys
from main_files.exception.exception import CropFertilizerException

class AdvisoryModel:
    """
    A scalable wrapper to combine preprocessing + model for multiple advisory tasks.
    Example tasks: 'crop', 'fertilizer', 'pest' (future).
    """

    def __init__(self):
        try:
            # Dictionary to store models for each advisory type
            self.models = {}
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def register_model(self, task_name: str, preprocessor, model):
        """
        Register a new advisory model with its preprocessor.
        Example: advisory.register_model("crop", crop_preprocessor, crop_model)
        """
        try:
            self.models[task_name] = {
                "preprocessor": preprocessor,
                "model": model
            }
        except Exception as e:
            raise CropFertilizerException(e, sys)

    def predict(self, task_name: str, x):
        """
        Predict for a given task (e.g., 'crop' or 'fertilizer').
        Ensures preprocessing is applied before prediction.
        """
        try:
            if task_name not in self.models:
                raise ValueError(f"No model registered for task: {task_name}")

            preprocessor = self.models[task_name]["preprocessor"]
            model = self.models[task_name]["model"]

            x_transform = preprocessor.transform(x)
            y_hat = model.predict(x_transform)
            return y_hat

        except Exception as e:
            raise CropFertilizerException(e, sys)

    def list_tasks(self):
        """Return all available registered tasks (e.g., ['crop', 'fertilizer'])."""
        return list(self.models.keys())
