import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, features_df):
        """Loads serialized assets and generates live sales forecasts"""
        try:
            logger.info("Loading inference assets for live prediction request...")
            
            # Load the Champion LightGBM model binary via utils
            model = load_object(file_path=self.model_path)
                
            # Generate the inference calculation
            prediction = model.predict(features_df)
            return prediction

        except Exception as e:
            logger.error("Exception encountered during operational inference cycle.")
            raise CustomException(e, sys)


class CustomData:
    """Maps individualized web application form inputs into model-ready features"""
    def __init__(
        self,
        store: int,
        date_str: str,  # Format: YYYY-MM-DD from HTML5 date picker
        holiday_flag: int,
        temperature: float,
        fuel_price: float,
        cpi: float,
        unemployment: float,
        sales_lag_1: float,
        sales_lag_2: float,
        sales_lag_4: float,
        rolling_mean_4: float,
        rolling_std_4: float,
        ema_4: float
    ):
        self.store = store
        self.date_str = date_str
        self.holiday_flag = holiday_flag
        self.temperature = temperature
        self.fuel_price = fuel_price
        self.cpi = cpi
        self.unemployment = unemployment
        self.sales_lag_1 = sales_lag_1
        self.sales_lag_2 = sales_lag_2
        self.sales_lag_4 = sales_lag_4
        self.rolling_mean_4 = rolling_mean_4
        self.rolling_std_4 = rolling_std_4
        self.ema_4 = ema_4

    def get_data_as_data_frame(self):
        """Applies feature transformations to match the production training feature space"""
        try:
            # 1. Parse operational date attributes
            date_obj = pd.to_datetime(self.date_str)
            year = date_obj.year
            month = date_obj.month
            day_of_week = date_obj.dayofweek
            week_of_year = int(date_obj.isocalendar()[1])

            # 2. Extract engineered holiday features
            super_bowl_week = 1 if (month == 2 and self.holiday_flag == 1) else 0
            labor_day_week = 1 if (month == 9 and self.holiday_flag == 1) else 0
            thanksgiving_week = 1 if (month == 11 and self.holiday_flag == 1) else 0
            christmas_week = 1 if (month == 12 and self.holiday_flag == 1) else 0

            # 3. Compute cyclical trigonometric features
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            week_sin = np.sin(2 * np.pi * week_of_year / 52)
            week_cos = np.cos(2 * np.pi * week_of_year / 52)

            # 4. Compute macroeconomic interaction feature
            fuel_unemployment_interaction = self.fuel_price * self.unemployment

            # 5. Load the un-leaked historical store baseline sales map via utils
            meta_path = os.path.join("artifacts", "store_avg_sales_meta.pkl")
            store_avg_sales_map = load_object(file_path=meta_path)
            
            # Map the baseline sales or fallback to the network average if it's a new store ID
            store_avg_sales = store_avg_sales_map.get(self.store, np.mean(list(store_avg_sales_map.values())))

            # 6. Construct the feature dictionary aligned *exactly* with the model's training columns
            input_data = {
                "Store": [self.store],
                "Holiday_Flag": [self.holiday_flag],
                "Temperature": [self.temperature],
                "Fuel_Price": [self.fuel_price],
                "CPI": [self.cpi],
                "Unemployment": [self.unemployment],
                "Year": [year],
                "Day_of_Week": [day_of_week],
                "Super_Bowl_Week": [super_bowl_week],
                "Labor_Day_Week": [labor_day_week],
                "Thanksgiving_Week": [thanksgiving_week],
                "Christmas_Week": [christmas_week],
                "Month_Sin": [month_sin],
                "Month_Cos": [month_cos],
                "Week_Sin": [week_sin],
                "Week_Cos": [week_cos],
                "Sales_Lag_1": [self.sales_lag_1],
                "Sales_Lag_2": [self.sales_lag_2],
                "Sales_Lag_4": [self.sales_lag_4],
                "Rolling_Mean_4": [self.rolling_mean_4],
                "Rolling_STD_4": [self.rolling_std_4],
                "EMA_4": [self.ema_4],
                "Fuel_Unemployment_Interaction": [fuel_unemployment_interaction],
                "Store_Avg_Sales": [store_avg_sales]
            }

            features_df = pd.DataFrame(input_data)
            logger.info("Successfully converted incoming request payload into model-ready DataFrame row.")
            return features_df

        except Exception as e:
            logger.error("Exception encountered during user input mapping conversion.")
            raise CustomException(e, sys)