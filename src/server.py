from flask import Flask, request, jsonify
import pandas as pd
from model_output import calculate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

app = Flask(__name__)

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['latitude', 'longitude', 'depth', 'magnitude']].values
    y = df['magnitude'].values
    return X, y

@app.route('/predict', methods=['GET'])
def predict():
    # Get parameters from the request
    
    # Create a dataframe
    X, y = load_data("../data/processed/processed_earthquake_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    scaler = StandardScaler()
    # Call the calculate function
    df = pd.read_csv("../data/processed/processed_earthquake_data.csv")
    X = df[['latitude', 'longitude', 'depth', 'magnitude']].values
    y_import = df[['magnitude']]
    y = calculate(X_test)
    print(X_test)
    
    y = pd.DataFrame(y, columns=['magnitude'])
    scaler.fit(y_import[['magnitude']])
    y_new = scaler.inverse_transform(y[[ 'magnitude']])
    # Return the result as JSON
    y_res=y_new.tolist()
    res=[]
    print(X_test)
    for i in range(len(y_res)):
        if(y_res[i][0]>2):
            res.append((X_test[i][0][0]*29.45+3.69,X_test[i][0][1]*124.365+16.769))
    return jsonify({'The following latitudes longitudes may experience earthquakes': res})

if __name__ == '__main__':
    app.run(debug=True)