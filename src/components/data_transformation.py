import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Default path where the fitted preprocessor object will be saved as a pickle file
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        # Load config so we know where to persist the preprocessor later
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Builds and returns a ColumnTransformer that preprocesses both numerical
        and categorical features.

        Numerical pipeline:
            1. SimpleImputer  – fills missing values with the column median
            2. StandardScaler – standardises values to zero-mean, unit-variance

        Categorical pipeline:
            1. SimpleImputer    – fills missing values with the most-frequent category
            2. OneHotEncoder    – converts each category to a binary column
            3. StandardScaler   – scales OHE output (with_mean=False to avoid
                                  breaking the sparse matrix from OHE)

        Returns:
            preprocessor (ColumnTransformer): unfitted transformer ready to be
                applied to training/test data.

        Raises:
            CustomException: wraps any underlying exception with traceback info.
        """
        try:
            # Features that are already numeric — only need imputation + scaling
            numerical_columns = ["writing_score", "reading_score"]

            # Features that are strings — need imputation, encoding, then scaling
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # --- Numerical pipeline ---
            num_pipeline= Pipeline(
                steps=[
                # Step 1: Replace NaNs with the median of each column
                ("imputer",SimpleImputer(strategy="median")),
                # Step 2: Scale so every feature has mean=0 and std=1
                ("scaler",StandardScaler())
                ]
            )

            # --- Categorical pipeline ---
            cat_pipeline=Pipeline(
                steps=[
                # Step 1: Replace NaNs with the most common category in that column
                ("imputer",SimpleImputer(strategy="most_frequent")),
                # Step 2: Convert each category value to its own 0/1 binary column
                ("one_hot_encoder",OneHotEncoder()),
                # Step 3: Scale the OHE output; with_mean=False keeps the sparse
                #         matrix representation intact (can't center a sparse matrix)
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into one transformer that routes each column
            # group through its dedicated pipeline
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        """
        Reads train/test CSVs, fits the preprocessor on training data,
        transforms both splits, appends the target column, and saves
        the fitted preprocessor to disk.

        Args:
            train_path (str): File path to the training CSV.
            test_path  (str): File path to the test CSV.

        Returns:
            tuple:
                - train_arr (np.ndarray): Transformed training features + target
                  concatenated as the last column.
                - test_arr  (np.ndarray): Transformed test features + target
                  concatenated as the last column.
                - preprocessor_obj_file_path (str): Path where the fitted
                  preprocessor pickle was saved.

        Raises:
            CustomException: wraps any underlying exception with traceback info.
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Build the unfitted ColumnTransformer
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"        # Column we want to predict
            numerical_columns = ["writing_score", "reading_score"]

            # Separate features (X) from target (y) for the training set
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # Separate features (X) from target (y) for the test set
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # fit_transform on train: learns statistics (median, categories, etc.)
            # and transforms in one step — prevents data leakage from test set
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            # transform only on test: applies the statistics learned from train
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # np.c_ horizontally stacks arrays column-wise.
            # Here we append the target column back to the right of the features.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Persist the fitted preprocessor so it can be reused during inference
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
