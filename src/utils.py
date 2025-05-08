import os
import sys

import numpy as np 
import pandas as pd
import pickle
import math
import gensim
import re
import tqdm

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException

from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            # y_train_pred = model.predict(X_train)
            y_pred = model.predict(X_test)
            # train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_pred)
            report[list(models.keys())[i]]=test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def clean_text(raw_text):
    corpus = []
    sw_list = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for i in range(0,len(raw_text)):
        review = re.sub('[^a-zA-Z]0-9', ' ',raw_text[i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in sw_list]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def avg_word2vec(doc, model, vector_size=100):
    # model = gensim.models.Word2Vec(doc, window=5, min_count=2)
    valid_words = [word for word in doc if word in model.wv.index_to_key]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)

def word_arr(a_words):
    X = []
    for i in tqdm(range(len(a_words))):
        X.append(avg_word2vec(a_words[i]))
    return X

def Word_token(a_corpus):
    words = []
    for sent in a_corpus:
        sent_token = sent_tokenize(sent)
        # for sent in sent_token:
        words.append(simple_preprocess(sent))
        # words_fixed = [' '.join(word) for word in words]
    # print(words)
    # words.reshape(-1,1)
    # print(words)
    return words

class AvgWord2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, vector_size=100):
        self.model = model
        self.vector_size = vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._avg_word2vec(doc) for doc in X])

    def _avg_word2vec(self, doc):
        # a_list = []
        valid_words = [word for word in doc if word in self.model.wv.index_to_key]
        if not valid_words:
            return np.zeros(self.vector_size)
        return(np.mean([self.model.wv[word] for word in valid_words], axis=0))
        # a_list.append(np.mean([self.model.wv[word] for word in valid_words], axis=0))
        # return a_list
