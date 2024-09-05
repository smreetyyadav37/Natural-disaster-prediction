import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

if __name__ == "__main__":
    input_shape = (None, 4)  # Adjust input shape as per your data dimensions
    model = build_model(input_shape)
    model.summary()
    model.save("result_data/model_architecture.h5")
    print("Model architecture saved.")