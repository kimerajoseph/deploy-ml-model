#from crypt import methods
from flask import Flask,request, url_for, redirect, render_template, jsonify
import os
from os.path import join, dirname, realpath

from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

model=load_model('my_second_pipeline')

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    col = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    data_unseen = pd.DataFrame([final], columns = col)
    #print(int_features)
    #print(final)
    prediction=predict_model(model, data=data_unseen, round = 0)
    prediction=int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be ${}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

@app.route('/predict_file',methods=['POST'])
def predict_csv():
    '''
    Predict using CSV file
    '''
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)
        # save the file
    col = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    # Use Pandas to parse the CSV file
    data_unseen = pd.read_csv(file_path)

    data_unseen_cols = []
    for column in data_unseen.columns:
        data_unseen_cols.append(column)

    # check and compare csv files
    if data_unseen_cols != col:
        return_error = f"wrong csv file. The csv should have {col} as columns"
        return render_template('home.html',error=return_error)
    
    else:
        prediction=predict_model(model, data=data_unseen)
        prediction[['bmi','Label']] = prediction[['bmi','Label']].round(decimals=1)
        #print(prediction)   
        
        return render_template('table.html',table=prediction.to_html(classes='table table-striped'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)