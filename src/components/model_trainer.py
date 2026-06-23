import os
import sys
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from src.exception import CustomException
from src.logger import logger

@dataclass
class ModelTrainerConfig:
    """Configuration class containing paths for the trained model artifact"""
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_path, test_path):
        """Loads processed data, trains the champion LightGBM model, and exports the binary"""
        logger.info("Initializing Model Trainer process stage...")
        try:
            # 1. Load the pre-processed data streams
            logger.info("Loading transformed training and testing matrices...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 2. Isolate independent features from target label (Weekly_Sales)
            # 'Store' is kept as a feature since it serves as a crucial categorical tracking ID
            X_train = train_df.drop(columns=['Weekly_Sales'])
            y_train = train_df['Weekly_Sales']
            X_test = test_df.drop(columns=['Weekly_Sales'])
            y_test = test_df['Weekly_Sales']

            logger.info(f"Feature space array locked with {X_train.shape[1]} input dimensions.")

            # 3. Instantiate and fit your fine-tuned LightGBM Regressor
            logger.info("Fitting production-ready LightGBM Regressor...")
            lgb_model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            lgb_model.fit(X_train, y_train)
            logger.info("Model fitting operations completed successfully.")

            # 4. Generate forecasts and extract metrics over our unseen validation split
            logger.info("Evaluating forecasting metrics over unseen future timeline split...")
            predictions = lgb_model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
            mape = mean_absolute_percentage_error(y_test, predictions) * 100
            r2 = r2_score(y_test, predictions)

            # Log insights into our structured text files
            logger.info("--- 📊 Production Model Pipeline Performance Summary ---")
            logger.info(f"Target R² (Variance Explained): {r2 * 100:.2f}%")
            logger.info(f"Mean Absolute Error (MAE): ${mae:,.2f}")
            logger.info(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
            logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

            # Quality Check: Fail if performance falls below our strict 85% R² validation threshold
            if r2 < 0.85:
                raise CustomException("Trained production model R² score drops below established 85% threshold bounds.", sys)

            # 5. Serialize and export the champion model binary
            logger.info(f"Serializing champion model artifact to: {self.model_trainer_config.trained_model_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            
            with open(self.model_trainer_config.trained_model_path, "wb") as file_obj:
                pickle.dump(lgb_model, file_obj)

            logger.info("Model Trainer component workflow completed successfully.")
            return r2

        except Exception as e:
            logger.error("Exception encountered within Model Trainer execution cycle.")
            raise CustomException(e, sys)