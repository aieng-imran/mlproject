import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion file paths.

    Uses @dataclass so these fields are auto-initialized without writing
    a manual __init__. All paths point to the 'artifacts/' folder, which
    acts as the output directory for saved data files.
    """
    # Path where the 80% training split will be saved
    train_data_path: str = os.path.join('artifacts', "train.csv")
    # Path where the 20% test split will be saved
    test_data_path: str = os.path.join('artifacts', "test.csv")
    # Path where a copy of the original raw dataset will be saved
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
    Handles reading the raw dataset from disk, saving it to the artifacts
    directory, and splitting it into train and test sets.
    """

    def __init__(self):
        """
        Initializes DataIngestion with a DataIngestionConfig instance,
        which holds all the file paths needed during ingestion.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw CSV dataset, saves it as-is to the artifacts folder,
        splits it into train/test sets, saves both splits, and returns
        their file paths for the next pipeline stage (data transformation).

        Returns:
            tuple: (train_data_path, test_data_path) — file paths to the
                   saved train and test CSV files inside 'artifacts/'.

        Raises:
            CustomException: Wraps any exception with traceback info for
                             consistent error reporting across the pipeline.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the raw student performance dataset into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the 'artifacts/' directory if it doesn't already exist;
            # exist_ok=True prevents an error if the folder is already there
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save a copy of the original raw data before any splitting
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the data: 80% for training, 20% for testing;
            # random_state=42 ensures reproducibility across runs
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Persist the training split to disk
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Persist the test split to disk
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths so downstream components can read the files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Wrap the original exception with file/line context via CustomException
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Step 1: Ingest raw data and produce train/test CSV files
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data saved to: {train_data}")
    print(f"Test data saved to:  {test_data}")

    # Step 2 & 3 are commented out until data_transformation.py and model_trainer.py are implemented
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))



