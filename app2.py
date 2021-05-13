# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:40:30 2021

@author: jacmckenna
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials
from functools import reduce
import csv
import json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    authoringKey = '20dc03f3f35e418b9c38bb1553ad8b24'
    authoringEndpoint = 'https://jemdemo5.cognitiveservices.azure.com/'
    predictionKey = '20dc03f3f35e418b9c38bb1553ad8b24'
    predictionEndpoint = 'https://jemdemo5.cognitiveservices.azure.com/'
    
    appName = "Luis Demo v20"
    versionId = "0.1"
    client = LUISAuthoringClient(authoringEndpoint, CognitiveServicesCredentials(authoringKey))
    app_id = '5eab7501-e61b-4b51-8831-d9f7657ee00f'
    
    responseEndpointInfo = client.apps.publish(app_id, versionId, is_staging=False)
    
    runtimeCredentials = CognitiveServicesCredentials(predictionKey)
    clientRuntime = LUISRuntimeClient(endpoint=predictionEndpoint, credentials=runtimeCredentials)

	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        my_prediction = clientRuntime.prediction.get_slot_prediction(app_id, "Production", {"query":message}).prediction.top_intent
        my_prediction = my_prediction.replace("_"," ")
        my_prediction = my_prediction.capitalize()
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)