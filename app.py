from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.utils import load_object
from src.exception import CustomException
import pickle

application = Flask(__name__)

app = application

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])


def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            message= request.form.get('message'),
            # bathrooms= float(request.form.get('bathrooms')),
            # location= request.form.get('location')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(type(results[0]))
        res_ult = results[0].tobytes()
        print(res_ult)
        res_ult_int = int.from_bytes(res_ult, byteorder='little')
        
        my_prediction = ''
        if res_ult_int == 1:
            my_prediction = 'Spam'
        else:
            my_prediction = 'Ham'
        print(f'my_prediction {my_prediction}')
        # return render_template('home.html', results = f"{pred_df['message'][0]}. \nHam or Spam: {my_prediction}")#{results[0]}
        return render_template('home.html', results =  my_prediction)#{results[0]}
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)