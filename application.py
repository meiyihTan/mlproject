#same as app.py, just rename to application.py for deployment 
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__) #entry point

app=application

# Route for a home page

@app.route('/')
def index():
    return render_template('index.html') #default home page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html') #the simple input data field that we need to provide to model for prediction
    else: #POST ; capture data, do standard, feature scaling, prediction
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score'))

        pred_df=data.get_data_as_data_frame() #get the input data in df form
        print(pred_df)
        print("Before Prediction")

        prediction_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=prediction_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0") #open site at 127.0.0.1:5000