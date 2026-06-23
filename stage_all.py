import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logger

if __name__ == "__main__":
    try:
        logger.info("==================================================")
        logger.info("Initiating End-to-End Core Production Pipeline Run")
        logger.info("==================================================")
        
        # 1. Trigger File Ingestion Pipeline Stage
        # Targets the baseline dataset we uploaded to notebook/data/Walmart.csv
        ingestion = DataIngestion()
        raw_data_path = ingestion.initiate_data_ingestion(input_csv_path="notebook/data/Walmart.csv")
        
        # 2. Trigger Feature Engineering & Pre-processing Pipeline Stage
        transformation = DataTransformation()
        train_feat_path, test_feat_path = transformation.initiate_data_transformation(raw_path=raw_data_path)
        
        # 3. Trigger Model Training, Quality Validation & Serialization Engine
        trainer = ModelTrainer()
        final_variance_score = trainer.initiate_model_trainer(train_path=train_feat_path, test_path=test_feat_path)
        
        print(f"\nPipeline Run Complete! Model Variance Explanation (R²): {final_variance_score * 100:.2f}%")
        logger.info("==================================================")
        logger.info("🎉 End-to-End Core Pipeline Completed Successfully!")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Pipeline Execution Aborted: {str(e)}")
        print(f"\nPipeline Failed. Check the latest log file in logs/ for structural traceback details.")
        sys.exit(1)