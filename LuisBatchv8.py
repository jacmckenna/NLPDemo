# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:35:33 2021

@author: jacmckenna
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:42:44 2021

@author: jacmckenna
"""
from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials
from functools import reduce
import csv
import json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def quickstart():

    authoringKey = '20dc03f3f35e418b9c38bb1553ad8b24'
    authoringEndpoint = 'https://jemdemo5.cognitiveservices.azure.com/'
    predictionKey = '20dc03f3f35e418b9c38bb1553ad8b24'
    predictionEndpoint = 'https://jemdemo5.cognitiveservices.azure.com/'
    
    appName = "Luis Demo v20"
    versionId = "0.1"
    
    client = LUISAuthoringClient(authoringEndpoint, CognitiveServicesCredentials(authoringKey))
    
    # define app basics
    appDefinition = {
        "name": appName,
        "initial_version_id": versionId,
        "culture": "en-us"
    }

# create app
    app_id = client.apps.add(appDefinition)

# get app id - necessary for all other changes
    print("Created LUIS app with ID {}".format(app_id))
    
# read training data
    with open('Intentsubset1.csv', 'r') as fd:
        reader = csv.DictReader(fd)
        labeledExampleUtteranceWithMLEntity = list(reader)
        
# Get unique intents   
    all_intents = []
    for x in labeledExampleUtteranceWithMLEntity:
        all_intents.append(x['intentName'])
    unique_intents = list(set(all_intents))
    
    for x in unique_intents:
        client.model.add_intent(app_id, versionId, x)
   
    # Define labeled example

    print("Labeled Example Utterance:", labeledExampleUtteranceWithMLEntity[2:4])
    print(len(labeledExampleUtteranceWithMLEntity))
    df = pd.DataFrame(labeledExampleUtteranceWithMLEntity)
    X = df['text']
    Y = df['intentName']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify = Y)
    print('finished split')
# Add an example for the entity.
# Enable nested children to allow using multiple models with the same name.
# The quantity subentity and the phraselist could have the same exact name if this is set to True
    chunks = (len(X_train) - 1) // 50 + 1
    Train_df = pd.concat([X_train, Y_train], axis = 1, join = "inner")
    Train_dict = Train_df.to_dict('records')
    print(len(Train_df))
    for i in range(chunks):
        client.examples.batch(app_id, versionId, Train_dict[i*50:(i+1)*50], { "enableNestedChildren": True })
    
    print('added examples')
    client.train.train_version(app_id, versionId)
    waiting = True
    while waiting:
        info = client.train.get_status(app_id, versionId)

    # get_status returns a list of training statuses, one for each model. Loop through them and make sure all are done.
        waiting = any(map(lambda x: 'Queued' == x.details.status or 'InProgress' == x.details.status, info))
        if waiting:
            print ("Waiting 10 seconds for training to complete...")
            time.sleep(10)
        else: 
            print ("trained")
            waiting = False
            
    responseEndpointInfo = client.apps.publish(app_id, versionId, is_staging=False)
    
    runtimeCredentials = CognitiveServicesCredentials(predictionKey)
    clientRuntime = LUISRuntimeClient(endpoint=predictionEndpoint, credentials=runtimeCredentials)
    
    
    # Production == slot name
    predictionRequest = { "query" : "I forgot my password" }

    predictionResponse = clientRuntime.prediction.get_slot_prediction(app_id, "Production", predictionRequest)
    print("Top intent: {}".format(predictionResponse.prediction.top_intent))
    print("Sentiment: {}".format (predictionResponse.prediction.sentiment))
    print("Intents: ")

    for intent in predictionResponse.prediction.intents:
        print("\t{}".format (json.dumps (intent)))
    print("Entities: {}".format (predictionResponse.prediction.entities))

quickstart()