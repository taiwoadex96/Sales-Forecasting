import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logger

def save_object(file_path: str, obj):
    """Saves a binary serialization object safely to disk"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logger.info(f"Asset object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """Loads a binary serialization object safely from disk"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target object file path does not exist: {file_path}")
            
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)