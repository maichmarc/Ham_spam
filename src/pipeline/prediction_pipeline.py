import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import pickle



class CustomData:
    def __init__(self, message: str):#, bathrooms: float, location: str):
        self.message = message

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'message':[self.message],
                # 'Bathrooms': [self.bathrooms],
                # 'Location': [self.location]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # model_path = 'artifact\model.pkl'
            # preprocessor_path = 'artifact\preprocessor.pkl'
            with open('artifact/preprocessor.pkl', 'rb') as preprocessor_obj:
                preprocessor = pickle.load(preprocessor_obj)
            with open('artifact/model.pkl', 'rb') as model_obj:
                model = pickle.load(model_obj)
            # preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys) 