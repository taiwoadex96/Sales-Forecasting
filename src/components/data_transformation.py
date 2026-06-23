import sys
import os
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger

@dataclass
class DataTransformationConfig:
    """Configuration class containing paths for processed files and metadata mapping"""
    transformed_train_path: str = os.path.join('artifacts', 'train_feat.csv')
    transformed_test_path: str = os.path.join('artifacts', 'test_feat.csv')
    meta_mapping_path: str = os.path.join('artifacts', 'store_avg_sales_meta.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def run_feature_engineering(self, df):
        """Applies feature transformations to the dataset"""
        try:
            df = df.copy()
            # Parse Date string to datetime object
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            
            # Filter returns/anomalous entries (sales less than 0)
            df = df[df['Weekly_Sales'] >= 0]
            
            # Calendar Extractors
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            
            # Specific Holiday Extractors
            df['Super_Bowl_Week'] = ((df['Month'] == 2) & (df['Holiday_Flag'] == 1)).astype(int)
            df['Labor_Day_Week'] = ((df['Month'] == 9) & (df['Holiday_Flag'] == 1)).astype(int)
            df['Thanksgiving_Week'] = ((df['Month'] == 11) & (df['Holiday_Flag'] == 1)).astype(int)
            df['Christmas_Week'] = ((df['Month'] == 12) & (df['Holiday_Flag'] == 1)).astype(int)
            
            # Cyclical Trigonometric Wave Transformations
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['Week_Sin'] = np.sin(2 * np.pi * df['Week_of_Year'] / 52)
            df['Week_Cos'] = np.cos(2 * np.pi * df['Week_of_Year'] / 52)
            
            # Time Series Autoregressive Lags (Grouped by individual Store units)
            df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)
            df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
            df['Sales_Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
            df['Sales_Lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)
            
            # Rolling Statistics and Exponential Windows over past performance
            df['Rolling_Mean_4'] = df.groupby('Store')['Sales_Lag_1'].transform(lambda x: x.rolling(4).mean())
            df['Rolling_STD_4'] = df.groupby('Store')['Sales_Lag_1'].transform(lambda x: x.rolling(4).std())
            df['EMA_4'] = df.groupby('Store')['Sales_Lag_1'].transform(lambda x: x.ewm(span=4).mean())
            
            # Macroeconomic Environment Interaction Feature
            df['Fuel_Unemployment_Interaction'] = df['Fuel_Price'] * df['Unemployment']
            
            # Drop NaN rows resulting from shifts/rolling windows
            df = df.dropna().reset_index(drop=True)
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_path):
        """Processes raw data, executes temporal train/test splitting, and applies target mapping"""
        logger.info("Initializing Data Transformation process stage...")
        try:
            raw_df = pd.read_csv(raw_path)
            
            # Run feature engineering pipeline
            processed_df = self.run_feature_engineering(raw_df)
            
            # Chronological Splitting (Maintains zero-leakage test horizon)
            cutoff_date = pd.to_datetime('2012-05-01')
            train_split = processed_df[processed_df['Date'] < cutoff_date].copy()
            test_split = processed_df[processed_df['Date'] >= cutoff_date].copy()
            
            # Target Encoding Map Construction using only historical train metrics (Leakage Shield)
            store_avg_sales = train_split.groupby('Store')['Weekly_Sales'].mean()
            
            # Map historical target encoding back to both sets cleanly
            train_split['Store_Avg_Sales'] = train_split['Store'].map(store_avg_sales)
            test_split['Store_Avg_Sales'] = test_split['Store'].map(store_avg_sales)
            
            # Drop structural/redundant columns that aren't numeric features
            features_to_drop = ['Date', 'Month', 'Week_of_Year']
            train_split = train_split.drop(columns=features_to_drop)
            test_split = test_split.drop(columns=features_to_drop)
            
            # Save processed matrices as CSV files
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_path), exist_ok=True)
            train_split.to_csv(self.transformation_config.transformed_train_path, index=False)
            test_split.to_csv(self.transformation_config.transformed_test_path, index=False)
            
            # Serialize the metadata mapping dictionary to apply to single web forms later
            with open(self.transformation_config.meta_mapping_path, 'wb') as f:
                pickle.dump(store_avg_sales, f)
                
            logger.info("Features processed and data mappings serialized successfully.")
            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path
            )
            
        except Exception as e:
            logger.error("Exception encountered within Data Transformation execution cycle.")
            raise CustomException(e, sys)