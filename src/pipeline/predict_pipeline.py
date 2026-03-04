import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """Loads the saved model and preprocessor artifacts, then runs inference."""

    def __init__(self):
        """No state needed at construction time; artifacts are loaded on demand."""
        pass

    def predict(self, features):
        """Transform raw features and return model predictions.

        Args:
            features (pd.DataFrame): Raw input data with the same columns used
                                     during training (before any scaling/encoding).

        Returns:
            np.ndarray: Array of predicted math scores, one value per input row.

        Raises:
            CustomException: Wraps any error that occurs during loading or inference.
        """
        try:
            # Build paths to the serialized artifacts produced by the training pipeline
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            # Deserialize the trained model (e.g. best regressor found by GridSearchCV)
            model = load_object(file_path=model_path)
            # Deserialize the fitted preprocessor (ColumnTransformer with scaling + encoding)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Apply the same transformations used during training (scale numerics, encode categoricals)
            data_scaled = preprocessor.transform(features)

            # Run inference on the transformed data
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Captures a single student's input from the web form and converts it to a
    DataFrame that the prediction pipeline can consume."""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        """Store each form field as an instance attribute.

        Args:
            gender                    (str): Student gender (e.g. 'male', 'female').
            race_ethnicity            (str): Ethnic group category (e.g. 'group B').
            parental_level_of_education     : Highest education level of parent
                                             (e.g. "bachelor's degree").
            lunch                     (str): Lunch type ('standard' or 'free/reduced').
            test_preparation_course   (str): Whether the student completed a prep
                                             course ('completed' or 'none').
            reading_score             (int): Student's reading exam score (0–100).
            writing_score             (int): Student's writing exam score (0–100).
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """Pack the instance attributes into a single-row DataFrame.

        The column names must exactly match the feature names the preprocessor
        was fitted on during training, otherwise the transform step will fail.

        Returns:
            pd.DataFrame: One-row DataFrame ready to be passed to
                          PredictPipeline.predict().

        Raises:
            CustomException: Wraps any error during DataFrame construction.
        """
        try:
            # Wrap each value in a list so pandas creates a single-row DataFrame
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dict to a DataFrame — shape will be (1, 7)
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
