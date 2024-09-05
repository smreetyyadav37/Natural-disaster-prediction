import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def calculate(X_test):
    model = tf.keras.models.load_model("../result_data/trained_model.h5")
    
    # Split data into training and testing sets
   
    y_pred = model.predict(X_test)
    return y_pred

