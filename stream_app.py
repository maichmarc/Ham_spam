import streamlit as st
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

st.title('Spam or Ham')
st.write("This is Natural Language Processing web app uses data from SMS Spam Collection dataset from UC Irvine Machine Learning Repository " \
"to create a Machine Learning algorithm to wheather the content of an SMS message is 'Spam' or 'Ham'. " \
"The data was trained using Word2Vec NLP algorithm and a Classifier applied to perform predictions.")


user_input = st.text_area('Enter Text to Analyse')
button = st.button("Analyse")


if user_input and button :

    data = CustomData(
        message= user_input,
        )
    pred_df = data.get_data_as_dataframe()
    print(pred_df)

 
    obj = PredictPipeline()
    text = obj.predict(pred_df)
    # results = predict_pipeline.predict(pred_df)
    # print(type(text[0]))
    res_ult = text[0].tobytes()
    # print(res_ult)
    res_ult_int = int.from_bytes(res_ult, byteorder='little')
    my_prediction = ''
    if res_ult_int == 1:
        my_prediction = 'Spam'
    else:
        my_prediction = 'Ham'

    st.write("Prediction: ", my_prediction)