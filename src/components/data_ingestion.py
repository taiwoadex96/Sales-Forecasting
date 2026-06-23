import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger  # Importing your clean, named logger instance

@dataclass
class DataIngestionConfig:
    """Configuration class containing artifact file paths for data ingestion"""
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, input_csv_path="notebook/data/Walmart.csv"):
        """
        Ingests data from the localized data directory, backups the raw stream, 
        and registers paths for subsequent data processing.
        """
        logger.info("Initializing Data Ingestion process stage...")
        try:
            # Check if input path exists to fail fast with a descriptive log
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Source file not found at path: {os.path.abspath(input_csv_path)}")

            # Read the base dataset
            df = pd.read_csv(input_csv_path)
            logger.info(f"Successfully loaded raw dataset from '{input_csv_path}' with shape: {df.shape}")

            # Ensure the output directory artifacts framework exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Export raw file replication backup into our artifacts workspace
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw source dataset securely mirrored in artifacts path: {self.ingestion_config.raw_data_path}")

            logger.info("Data Ingestion stage execution successfully completed.")
            return self.ingestion_config.raw_data_path

        except Exception as e:
            logger.error("Exception encountered within Data Ingestion execution cycle.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Smoke-test execution block to test ingestion isolatedly if run directly
    obj = DataIngestion()
    obj.initiate_data_ingestion()