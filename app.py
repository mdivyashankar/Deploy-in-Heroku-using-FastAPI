# Importing required libraries
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

app = FastAPI()					        ### Creating FastAPI application object/instance

# Reading the model pickle file
classifier = pickle.load(open("classifier.pkl","rb"))

# Defining the routes through the created object

# Opens automatically on http://127.0.0.1:8000
@app.get('/')                                           ### It indicates that the index function is responsible for handling requests that go to the endpoint “/” using the get operation.
def index():
    return {'message': 'Hello!'}

# Endpoint: http://127.0.0.1:8000/AnyNameHere   
@app.get('/{name}')				        ### Route with a single parameter, returns the parameter within a message
def get_name(name: str):
    return {'Welcome To ML Deployment with FastAPI': f'{name}'}

# Endpoint: http://127.0.0.1:8000/predict
@app.post('/predict')				        ### Expose the prediction functionality, make a prediction from the i/p JSON data & return the predicted Bank Note
def predict_banknote(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if (prediction[0] > 0.5):
        prediction = "Fake note"
    else:
        prediction = "Its a Bank note"
    return {'prediction': prediction}

# Will run on http://127.0.0.1:8000 
if __name__ == '__main__':				### On running python main.py, runs the FastAPI app
    uvicorn.run(app, host = '127.0.0.1', port = 8000)	### Run the FastAPI app with uvicorn