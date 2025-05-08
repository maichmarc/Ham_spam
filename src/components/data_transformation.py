import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
from src.utils import clean_text, avg_word2vec, word_arr, Word_token
from src.utils import save_object
from scipy.sparse import csr_matrix, hstack

@dataclass
class DataTransformationConfig():
    new_data_path: str = os.path.join('artifact', 'new_raw.csv')
    # y_data_path: str = os.path.join('artifact', 'y.csv')
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class AvgWord2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, vector_size=100):
        self.model = model
        self.vector_size = vector_size

    def fit(self, X, y=None):
        return self

    # def transform(self, X):
    #     vectors = np.array([avg_word2vec(doc, self.model, self.vector_size) for doc in X])
    #     print(f'shape of output vector array: {vectors.shape}')
    #     return vectors
    
    def transform(self, X):
        vectors = clean_text(X)
        vectors = Word_token(X)
        vectors = np.array([avg_word2vec(doc, self.model, self.vector_size) for doc in X])
        print(f'shape of output vector array: {vectors.shape}')
        return vectors

    
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)

class WordTokeniser(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            df = pd.read_csv('artifact/raw.csv') 
                    
            logging.info('Read the dataset as dataframe')
            X = df['message']
            y = df['label']
           
            model = gensim.models.Word2Vec(X, window=5, min_count=2)
            new_raw = pd.DataFrame({
                'message': X,
                'label': y
            })
            new_raw.to_csv(self.data_transformation_config.new_data_path, index=False, header=True)
           

            preprocessor = ColumnTransformer([
                ('Avg_w2v',AvgWord2VecTransformer(model=model, vector_size=100), 'message')
            ])

           
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):#, X_data_path, y_data_path):

        try:
            df = pd.read_csv('artifact/new_raw.csv')
            # df = pd.read_csv('D:/PROJECTS/Spam_ham/artifact/new_raw.csv')
            logging.info('Read the preprocessed dataset as dataframe')
            # print(df.head())
            
            X = df[['message']]
            y = df['label']
         

            logging.info('Reading X and y data completed.')
            logging.info('Obtaining preprocessing object.')

            preprocessor_obj = self.get_data_transformer_obj()

            logging.info('Applying preprocessor object on training and testing dataframe')
                       
            X_array = preprocessor_obj.fit_transform(X)
            

            encoder = LabelEncoder()

            y_array = encoder.fit_transform(y)

            print(type(X_array))
            print(f'X_array shape: {X_array.shape}')
            print(X_array[1])
            print(type(y_array))
            print(f'y_array shape: {y_array.shape}')
            print(y_array[0:10])



            logging.info('Saved preprocessing object.')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                X_array,
                y_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)   