import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(
            "Entered the Data Ingestion method or component"
        )

        try:

            # Get current directory
            current_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

            # Go to project root
            project_root = os.path.abspath(
                os.path.join(current_dir, "..", "..")
            )

            # Dataset path
            data_path = os.path.join(
                project_root,
                "notebook",
                "data",
                "stud.csv"
            )

            print("Dataset Path:", data_path)

            # Read CSV file
            df = pd.read_csv(data_path)

            logging.info(
                "Read the dataset as dataframe"
            )

            # Create artifacts directory
            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Train test split initiated"
            )

            # Split data
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save test data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Ingestion of the data is completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    # Data Ingestion
    obj = DataIngestion()

    train_data, test_data = (
        obj.initiate_data_ingestion()
    )

    # Data Transformation
    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    # Model Training
    modeltrainer = ModelTrainer()

    print(
        modeltrainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
    )