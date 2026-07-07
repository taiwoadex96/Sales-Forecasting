import os
import sys
import math
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
            model = load_object(file_path=self.model_path)
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
        date_str: str,
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
        self.input_rolling_std_val = rolling_std_4
        self.ema_4 = ema_4

    def get_data_as_data_frame(self):
        """Applies feature transformations cleanly and maps data types explicitly"""
        try:
            # 1. Parse operational date attributes
            date_obj = pd.to_datetime(self.date_str)
            year = int(date_obj.year)
            month = int(date_obj.month)
            day_of_week = int(date_obj.dayofweek)
            week_of_year = int(date_obj.isocalendar()[1])

            # 2. Extract engineered holiday features
            super_bowl_week = 1 if (month == 2 and self.holiday_flag == 1) else 0
            labor_day_week = 1 if (month == 9 and self.holiday_flag == 1) else 0
            thanksgiving_week = 1 if (month == 11 and self.holiday_flag == 1) else 0
            christmas_week = 1 if (month == 12 and self.holiday_flag == 1) else 0

            # 3. Compute cyclical wave features natively using math rules
            month_sin = float(math.sin(2 * math.pi * month / 12))
            month_cos = float(math.cos(2 * math.pi * month / 12))
            week_sin = float(math.sin(2 * math.pi * week_of_year / 52))
            week_cos = float(math.cos(2 * math.pi * week_of_year / 52))

            # 4. Compute macroeconomic interaction feature
            fuel_unemployment_interaction = float(self.fuel_price * self.unemployment)

            # 5. Load target baseline historical metadata series safely
            meta_path = os.path.join("artifacts", "store_avg_sales_meta.pkl")
            store_avg_sales_series = load_object(file_path=meta_path)
            
            # Extract historical baseline or calculate global mean fallback safely from the Series
            global_mean_fallback = float(store_avg_sales_series.mean())
            
            if self.store in store_avg_sales_series.index:
                store_avg_sales = float(store_avg_sales_series.loc[self.store])
            else:
                store_avg_sales = global_mean_fallback

            # 6. Construct dictionary reflecting exact column configurations from model training
            input_data = {
                "Store": [int(self.store)],
                "Holiday_Flag": [int(self.holiday_flag)],
                "Temperature": [float(self.temperature)],
                "Fuel_Price": [float(self.fuel_price)],
                "CPI": [float(self.cpi)],
                "Unemployment": [float(self.unemployment)],
                "Year": [int(year)],
                "Day_of_Week": [int(day_of_week)],
                "Super_Bowl_Week": [int(super_bowl_week)],
                "Labor_Day_Week": [int(labor_day_week)],
                "Thanksgiving_Week": [int(thanksgiving_week)],
                "Christmas_Week": [int(christmas_week)],
                "Month_Sin": [month_sin],
                "Month_Cos": [month_cos],
                "Week_Sin": [week_sin],
                "Week_Cos": [week_cos],
                "Sales_Lag_1": [float(self.sales_lag_1)],
                "Sales_Lag_2": [float(self.sales_lag_2)],
                "Sales_Lag_4": [float(self.sales_lag_4)],
                "Rolling_Mean_4": [float(self.rolling_mean_4)],
                "Rolling_STD_4": [float(self.input_rolling_std_val)],
                "EMA_4": [float(self.ema_4)],
                "Fuel_Unemployment_Interaction": [fuel_unemployment_interaction],
                "Store_Avg_Sales": [store_avg_sales]
            }

            features_df = pd.DataFrame(input_data)
            
            # Explicitly lock down column data types for LightGBM layout matching
            features_df['Store'] = features_df['Store'].astype('int64')
            features_df['Holiday_Flag'] = features_df['Holiday_Flag'].astype('int64')
            features_df['Year'] = features_df['Year'].astype('int64')
            features_df['Day_of_Week'] = features_df['Day_of_Week'].astype('int64')
            features_df['Super_Bowl_Week'] = features_df['Super_Bowl_Week'].astype('int64')
            features_df['Labor_Day_Week'] = features_df['Labor_Day_Week'].astype('int64')
            features_df['Thanksgiving_Week'] = features_df['Thanksgiving_Week'].astype('int64')
            features_df['Christmas_Week'] = features_df['Christmas_Week'].astype('int64')

            logger.info("Successfully converted incoming request payload into model-ready DataFrame row.")
            return features_df

        except Exception as e:
            logger.error("Exception encountered during user input mapping conversion.")
            raise CustomException(e, sys)