from flask import Flask, render_template,url_for,flash,redirect,request
import sklearn
import pickle
from flask import Flask, render_template
import xgboost
import json
from werkzeug.utils import secure_filename
import pandas as pd
import os
import numpy as np
from io import BytesIO




app = Flask(__name__)
model = pickle.load(open('dprofiling.pkl','rb'))
modelpmsm = pickle.load(open('DecisionTreeRegressor_model (1).pkl','rb'))
modelpm_range = pickle.load(open('BaggingRegressor_model_pm_range.pkl','rb'))
modelsoc = pickle.load(open('linear_regression_model2.pkl','rb'))


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
@app.route('/driverprofiling',methods=['GET','POST'])
def driverprofiling():
    return render_template('driverprofile.html',text_boxes=6)
@app.route('/pmsm',methods=['GET','POST'])
def pmsm():
    return render_template('pmsm.html',text_boxes=13)
@app.route('/socperc',methods=['GET','POST'])
def socperc():
    return render_template('Socperc.html',text_boxes=13)

@app.route('/predictdp',methods=['GET','POST'])
def predictdp():
    asensor_values = [float(request.form[f'asensor{i}']) for i in range(1, 4)]
    gsensor_values = [float(request.form[f'gsensor{i}']) for i in range(1, 4)]
    parameters = asensor_values+gsensor_values
    prediction_result = model.predict([parameters])
    return render_template('resultdp.html',prediction_result=prediction_result)

@app.route('/predictpmsm',methods=['GET','POST'])
def predictpmsm():
    parameters = []
    parameters_reshaped = []
    sensor_values1 = [17]
    sensor_values1 += [float(request.form[f'a{i}']) for i in range(1, 12)]  # Use += for list concatenation
    print(sensor_values1)
    pm_range = modelpm_range.predict([sensor_values1])
    print(pm_range[0])

    parameters = sensor_values1 + [pm_range[0]]  # Use + for list concatenation
    print(parameters, '\n')
    parameters_reshaped = np.array(parameters).reshape(1, -1)
    print(parameters_reshaped, '\n')  # Reshape to a 2D array
    prediction_result = modelpmsm.predict(parameters_reshaped)
    return render_template('resultpmsm.html', prediction_result=prediction_result)
@app.route('/socpredict',methods=['GET','POsT'])
def predictsoc():
    input_features = [float(request.form[f'b{i}']) for i in range (1,24)]
    prediction = modelsoc.predict([input_features])
    return render_template('resultsoc.html',prediction_result=prediction)


if __name__== '__main__':
    app.run(debug=True)

