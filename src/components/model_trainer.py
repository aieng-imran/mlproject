import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Holds file path configuration for saving the trained model artifact."""

    # Default path where the best trained model will be serialized as a .pkl file
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Trains multiple regression models, selects the best one via GridSearchCV,
    and persists it to disk."""

    def __init__(self):
        """Loads the path config so every method knows where to save the model."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Train and evaluate all candidate models, then save and return the best one.

        Args:
            train_array (np.ndarray): Combined array of training features + target.
                                      Last column is the target (y_train).
            test_array  (np.ndarray): Combined array of test features + target.
                                      Last column is the target (y_test).

        Returns:
            float: R² score of the best model evaluated on the test set.

        Raises:
            CustomException: If no model achieves an R² >= 0.6, or any other
                             unexpected error occurs during training.
        """
        try:
            logging.info("Split training and test input data")

            # Slice all columns except the last as features; last column is the label
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # training features
                train_array[:, -1],    # training labels
                test_array[:, :-1],    # test features
                test_array[:, -1],     # test labels
            )

            # --- Candidate models ---
            # Each model is instantiated with its default hyperparameters here;
            # the actual tuning happens inside evaluate_models via GridSearchCV.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),  # suppress CatBoost training logs
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # --- Hyperparameter search grids ---
            # For each model, list the values GridSearchCV will try for every
            # hyperparameter. An empty dict means "use defaults / no tuning".
            params = {
                "Decision Tree": {
                    # Try different split-quality criteria
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],      # commented out to keep search space small
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    # Number of trees to build — more trees = more stable but slower
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # Step size shrinkage used to prevent overfitting
                    'learning_rate': [.1, .01, .05, .001],
                    # Fraction of samples used per tree — adds randomness to reduce variance
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                # Linear Regression has no hyperparameters to tune
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],              # depth of each decision tree
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]        # number of boosting rounds
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
            }

            # evaluate_models runs GridSearchCV for each model and returns a
            # dict of {model_name: best_test_r2_score}
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models,   param=params,
            )

            # Pick the highest R² score across all models
            best_model_score = max(sorted(model_report.values()))

            # Retrieve the name that corresponds to that score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Fetch the (already-fitted) model object using the winning name
            best_model = models[best_model_name]

            # Reject the pipeline if no model is good enough (R² < 0.6 is too weak)
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # Persist the best model to disk so it can be loaded during inference
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Generate predictions on the held-out test set for final evaluation
            predicted = best_model.predict(X_test)

            # R² measures how well the model explains variance in the target
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
