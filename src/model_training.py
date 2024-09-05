import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['latitude', 'longitude', 'depth', 'magnitude']].values
    y = df['magnitude'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model):
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return history

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("data/processed/processed_earthquake_data.csv")
    
    model = tf.keras.models.load_model("result_data/model_architecture.h5")
    
    # Reshape X_train and X_test to match input shape (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    history = train_model(X_train, y_train, model)
    model.save("result_data/trained_model.h5")
    print("Model trained and saved.")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("result_data/training_history.csv", index=False)
    print("Training history saved.")