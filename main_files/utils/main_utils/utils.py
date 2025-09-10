import os
import sys
import yaml
import numpy as np
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from main_files.entity.artifact_entity import ClassificationMetricArtifact

from main_files.exception.exception import CropFertilizerException
from main_files.loggings.logger import logging

# YAML UTILS
def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    Used for schema.yaml or drift reports.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CropFertilizerException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary or object to a YAML file.
    Example: drift_report.yml
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise CropFertilizerException(e, sys)


# NUMPY UTILS
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array data to file (.npy).
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise CropFertilizerException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file (.npy).
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise CropFertilizerException(e, sys)


# OBJECT UTILS (pickle)
def save_object(file_path: str, obj: object) -> None:
    """
    Save Python object to a pickle file.
    Example: preprocessing.pkl, crop_model.pkl
    """
    try:
        logging.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise CropFertilizerException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load Python object from a pickle file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CropFertilizerException(e, sys)

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# def evaluate_models(x_train, y_train, x_test, y_test, models, params):
#     """
#     Run hyperparameter tuning (GridSearchCV) for multiple classification models,
#     retrain with best params, and compute metrics.
#     Returns a dictionary: 
#     {
#         model_name: {
#             best_model,
#             best_params,
#             train_metrics,
#             test_metrics,
#             primary_metric  # float for easy best model selection
#         }
#     }
#     """
#     try:
#         report = {}

#         for model_name, model in models.items():
#             param_grid = params.get(model_name, {})

#             # --- Grid search with f1_macro ---
#             gs = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
#             gs.fit(x_train, y_train)

#             # --- Retrain with best params ---
#             best_params = gs.best_params_
#             model.set_params(**best_params)
#             model.fit(x_train, y_train)

#             # --- Predictions ---
#             y_train_pred = model.predict(x_train)
#             y_test_pred = model.predict(x_test)

#             # --- Train metrics ---
#             train_metrics = ClassificationMetricArtifact(
#                 f1_score=f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
#                 precision_score=precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
#                 recall_score=recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
#             )

#             # --- Test metrics ---
#             test_metrics = ClassificationMetricArtifact(
#                 f1_score=f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
#                 precision_score=precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
#                 recall_score=recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
#             )

#             # --- Choose primary metric (for comparison across models) ---
#             primary_metric = test_metrics.f1_score  # weighted F1

#             report[model_name] = {
#                 "best_model": model,
#                 "best_params": best_params,
#                 "train_metrics": train_metrics,
#                 "test_metrics": test_metrics,
#                 "primary_metric": primary_metric,
#             }

#         return report

#     except Exception as e:
#         raise CropFertilizerException(e, sys)

# def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=3, scoring="f1_weighted"):
#     """
#     Evaluate multiple models with GridSearchCV + cross-validation.
#     Returns a dictionary with model_name -> results including:
#       - best_model (fitted on full train set with best params)
#       - best_params
#       - cv_score (mean cross-validation score)
#       - train_metrics
#       - test_metrics
#     """
#     try:
#         report = {}

#         for model_name, model in models.items():
#             param_grid = params.get(model_name, {})

#             # Grid search (with CV)
#             gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
#             gs.fit(X_train, y_train)

#             best_model = gs.best_estimator_
#             best_params = gs.best_params_
#             cv_score = gs.best_score_

#             # Predictions
#             y_train_pred = best_model.predict(X_train)
#             y_test_pred = best_model.predict(X_test)

#             # Train metrics
#             train_metrics = ClassificationMetricArtifact(
#                 f1_score=f1_score(y_train, y_train_pred, average="weighted"),
#                 precision_score=precision_score(y_train, y_train_pred, average="weighted"),
#                 recall_score=recall_score(y_train, y_train_pred, average="weighted"),
#             )

#             # Test metrics
#             test_metrics = ClassificationMetricArtifact(
#                 f1_score=f1_score(y_test, y_test_pred, average="weighted"),
#                 precision_score=precision_score(y_test, y_test_pred, average="weighted"),
#                 recall_score=recall_score(y_test, y_test_pred, average="weighted"),
#             )

#             report[model_name] = {
#                 "best_model": best_model,
#                 "best_params": best_params,
#                 "cv_score": cv_score,
#                 "train_metrics": train_metrics,
#                 "test_metrics": test_metrics,
#             }

#         return report

#     except Exception as e:
#         raise CropFertilizerException(e, sys)

from sklearn.model_selection import GridSearchCV
from main_files.exception.exception import CropFertilizerException
import sys

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    """
    Run hyperparameter tuning (GridSearchCV) for multiple models,
    retrain with best params, and compute metrics.
    Returns a dict {model_name: {...}} with cv_score, metrics, etc.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            # Grid search with CV
            gs = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
            gs.fit(x_train, y_train)

            best_params = gs.best_params_
            cv_score = gs.best_score_   # ✅ average CV score

            # Retrain on full training set with best params
            model.set_params(**best_params)
            model.fit(x_train, y_train)

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_metrics = ClassificationMetricArtifact(
                f1_score=f1_score(y_train, y_train_pred, average="weighted"),
                precision_score=precision_score(y_train, y_train_pred, average="weighted"),
                recall_score=recall_score(y_train, y_train_pred, average="weighted"),
            )

            test_metrics = ClassificationMetricArtifact(
                f1_score=f1_score(y_test, y_test_pred, average="weighted"),
                precision_score=precision_score(y_test, y_test_pred, average="weighted"),
                recall_score=recall_score(y_test, y_test_pred, average="weighted"),
            )

            report[model_name] = {
                "best_model": model,
                "best_params": best_params,
                "cv_score": cv_score,                     # ✅ add CV score
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "primary_metric": cv_score                 # ✅ use CV score for selection
            }

        return report

    except Exception as e:
        raise CropFertilizerException(e, sys)
