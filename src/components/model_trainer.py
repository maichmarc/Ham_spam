import os 
import sys

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, accuracy_score, confusion_matrix

from dataclasses import dataclass

from src.logger import logging
from src.utils import CustomException
from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_array, y_array):
        try:
            logging.info('Split training and test input data')
            X_train, X_test, y_train, y_test = train_test_split(X_array,y_array,test_size=0.2,random_state=42)
            print(f'X_train shape: {X_train.shape}')
            print(f'X_test shape: {X_test.shape}')
            print(f'y_train shape: {y_train.shape}')
            print(f'y_test shape: {y_test.shape}')
            
            models = {
            'Gaussian Naive-Bayes': GaussianNB(),
            'Decision Tree Classifier': DecisionTreeClassifier(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            # 'CatBoost Regressor Classifier': CatBoostClassifier(verbose=False),
            'AdaBoost Regressor Classifier': AdaBoostClassifier(),
            }

            model_report : dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # To get the best model score
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException('No best Model')
            
            print(model_report)
            
            logging.info(f'{best_model_name} is the best model with an r2_score of {best_model_score}.')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)