# =========================================
# LSTM for Cherry Blossom Bloom Forecasting
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow_datasets as tfds

from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. Load Inbuilt TFDS Dataset
# -------------------------------

dataset_tfds, info = tfds.load(
    "cherry_blossoms",
    split="train",
    with_info=True
)

# Convert to Pandas DataFrame
data = tfds.as_dataframe(dataset_tfds)

# Select required column and REMOVE NaN values
data = data[["doy"]].dropna()

dataset = data["doy"].values.astype("float32")

# -------------------------------
# 2. Visualize Dataset
# -------------------------------

plt.plot(dataset)
plt.xlabel("Year Index")
plt.ylabel("Day of Year (Bloom)")
plt.title("Cherry Blossom Bloom Timing")
plt.show()

# -------------------------------
# 3. Preprocessing
# -------------------------------

dataset = dataset.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size

train = dataset[:train_size]
test = dataset[train_size:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# -------------------------------
# 4. Time-Series Windowing
# -------------------------------

time_stamp = 10

def create_dataset(data, time_stamp):
    dataX, dataY = [], []
    for i in range(len(data) - time_stamp - 1):
        a = data[i:(i + time_stamp), 0]
        dataX.append(a)
        dataY.append(data[i + time_stamp, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, time_stamp)
testX, testY = create_dataset(test, time_stamp)

trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

# -------------------------------
# 5. LSTM Model
# -------------------------------

model = Sequential()
model.add(LSTM(10, input_shape=(1, time_stamp)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

model.summary()

# -------------------------------
# 6. Architecture Diagram
# -------------------------------

plot_model(model, show_shapes=True, show_layer_names=True)

# -------------------------------
# 7. Predictions
# -------------------------------

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))

# -------------------------------
# 8. RMSE Evaluation
# -------------------------------

trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))

print(f"Train RMSE: {trainScore:.2f}")
print(f"Test RMSE: {testScore:.2f}")

# -------------------------------
# 9. Plot Predictions
# -------------------------------

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stamp:len(trainPredict) + time_stamp, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (time_stamp * 2) + 1:len(dataset) - 1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset), label="Actual Values")
plt.plot(trainPredictPlot, label="Train Predictions")
plt.plot(testPredictPlot, label="Test Predictions")
plt.xlabel("Year Index")
plt.ylabel("Bloom Day")
plt.title("Cherry Blossom Bloom Forecasting using LSTM")
plt.legend()
plt.show()
