# predict.py
# Chet Russell

import pandas as pd
import xgboost as xgb
from numpy import array

model_path = "model_classification.json"

# loading model
model = xgb.Booster({"nthread": 4})  # init model

model.load_model(model_path)  # load model data

row_list = [1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,60.0]

row_array = array(row_list).reshape(1, -1)  # Reshape ensures it's 2D

# predicting...
pred = model.inplace_predict(row_array)
pred = pred.tolist()
# m = max(pred[0])
guess = pred[0]  # .index(pred)
print("Prediction: " + str(guess))