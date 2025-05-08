import os
import sys


from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import clean_text
from src.utils import avg_word2vec
from src.utils import word_arr
from src.utils import Word_token
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifact', 'raw.csv')
    # prep_data_path: str = os.path.join('artifact', 'prep.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion component')
        try:
            df=pd.read_csv('notebooks\data\SMSSpamCollection.txt', sep="\t", names=["label", "message"])
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Ingestion of the data is complete')
            return(
                self.ingestion_config.raw_data_path,
                # self.ingestion_config.prep_data_path, 
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    # X, y,_= data_transformation.get_data_transformer_obj()
    X_arr, y_arr,_ = data_transformation.initiate_data_transformation()

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_arr, y_arr))