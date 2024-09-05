import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['latitude', 'longitude', 'depth', 'magnitude']].values
    y = df['magnitude'].values
    return X, y

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

if __name__ == "__main__":
    X, y = load_data("data/processed/processed_earthquake_data.csv")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the model
    model = tf.keras.models.load_model("result_data/trained_model.h5")
    
    # Reshape X_test to match input shape (samples, time_steps, features)
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Evaluate the model
    mse, mae, r2 = evaluate_model(X_test, y_test, model)
    
    # Save evaluation metrics to file
    with open("result_data/evaluation.txt", "w") as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Mean Absolute Error: {mae}\n")
        f.write(f"R Squared: {r2}\n")
    
    print("Evaluation completed and results saved.")
    
    # Plot residuals
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='k')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residuals Histogram')
    plt.savefig("result_data/results/residuals_histogram.png")
    plt.close()

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    plt.savefig("result_data/results/actual_vs_predicted.png")
    plt.close()