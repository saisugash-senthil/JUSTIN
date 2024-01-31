from flask import Flask, render_template,url_for,flash,redirect,request
import sklearn
import pickle
from flask import Flask, render_template
import xgboost
import json
from werkzeug.utils import secure_filename
import pandas as pd
import os
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64




app = Flask(__name__)
model = pickle.load(open('dprofiling.pkl','rb'))
modelpmsm = pickle.load(open('DecisionTreeRegressor_model (1).pkl','rb'))


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
@app.route('/driverprofiling',methods=['GET','POST'])
def driverprofiling():
    return render_template('driverprofile.html',text_boxes=6)
@app.route('/pmsm',methods=['GET','POST'])
def pmsm():
    return render_template('pmsm.html',text_boxes=13)

@app.route('/predictdp',methods=['GET','POST'])
def predictdp():
    asensor_values = [float(request.form[f'asensor{i}']) for i in range(1, 4)]
    gsensor_values = [float(request.form[f'gsensor{i}']) for i in range(1, 4)]
    parameters = asensor_values+gsensor_values
    prediction_result = model.predict([parameters])
    return render_template('resultdp.html',prediction_result=prediction_result)

@app.route('/predictpmsm',methods=['GET','POST'])
def predictpmsm():
    sensor_values = {float(request.form[f'a{i}']) for i in range(0, 13)}
    parameters = sensor_values
    prediction_result = modelpmsm.predict([parameters])
    return render_template('resultpmsm.html',prediction_result=prediction_result)

if __name__== '__main__':
    app.run(debug=True)

