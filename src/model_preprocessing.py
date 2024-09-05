import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_earthquake_data(file_path):
    try:
        # Read the data
        df = pd.read_csv("C:/Users/smrit/Desktop/Coding/Machine_Learning/woodpecker/data/raw/earthquake_data.csv")
        print(f"Data loaded successfully from {'C:/Users/smrit/Desktop/Coding/Machine_Learning/woodpecker/data/raw/earthquake_data.csv'}")
        
        # Convert time to datetime and extract features
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        
        # Drop irrelevant columns
        df = df.drop(columns=['place'])
        print("Irrelevant columns dropped")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df[['latitude', 'longitude', 'depth', 'magnitude']] = imputer.fit_transform(df[['latitude', 'longitude', 'depth', 'magnitude']])
        print("Missing values handled")
        
        # Normalize features
        scaler = StandardScaler()
        df[['latitude', 'longitude', 'depth', 'magnitude']] = scaler.fit_transform(df[['latitude', 'longitude', 'depth', 'magnitude']])
        print("mean:"+str(scaler.mean_))
        sd=[]
        for var in scaler.var_:
            print("sd:"+str(math.sqrt(var)))

        
        # Save processed data
        df.to_csv("data/processed/processed_earthquake_data.csv", index=False)
        print("Earthquake data preprocessed and saved to processed_earthquake_data.csv")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function with the path to your data
preprocess_earthquake_data("C:/Users/smrit/Desktop/Coding/Machine_Learning/woodpecker/data/raw/earthquake_data.csv")