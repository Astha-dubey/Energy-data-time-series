import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageTk

# Load and preprocess your energy consumption data
# Replace this with your data loading code
data = pd.read_csv('energy_data.csv')
# Preprocess the data as needed

# Normalize the data
scaler = MinMaxScaler()
data['EnergyConsumption'] = scaler.fit_transform(data['EnergyConsumption'].values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Function to create sequences from data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length+1]
        sequences.append(seq)
    return np.array(sequences)

# Define the sequence length and features
sequence_length = 10
n_features = 1

train_sequences = create_sequences(train_data['EnergyConsumption'], sequence_length)
test_sequences = create_sequences(test_data['EnergyConsumption'], sequence_length)

# Split training sequences into X_train and y_train
X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]

# Build an LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (you may need to adjust epochs and batch size)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Create a Tkinter GUI
root = tk.Tk()
root.title("Energy Consumption Forecasting")

# Set a fixed window size (change width and height as needed)
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Load the background image and resize it to fit the window
background_image = Image.open('background.jpg')
background_image = background_image.resize((window_width, window_height), Image.BILINEAR)  # Use 'Image.BILINEAR' instead of 'Image.ANTIALIAS'
background_photo = ImageTk.PhotoImage(background_image)

# Create a label to display the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add input fields and widgets on top of the background
label = ttk.Label(root, text="Enter the Number of Days to Forecast:")
label.pack()

entry = ttk.Entry(root)
entry.pack()

# Define a function for making predictions
def make_forecast():
    try:
        # Get user input for the number of days to forecast
        horizon = int(entry.get())
        
        # Perform forecasting using the LSTM model
        # Replace this with your forecasting logic
        last_sequence = X_train[-1].reshape(1, sequence_length, n_features)
        forecast = []

        for i in range(horizon):
            prediction = model.predict(last_sequence)
            forecast.append(prediction[0, 0])
            last_sequence = np.append(last_sequence[:, 1:, :], prediction[0, 0].reshape(1, 1, 1), axis=1)

        # Inverse transform the forecasted values
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

        # Create an index for the forecast dates
        last_date = pd.to_datetime(data.index[-1])
        forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq='D')

        # Display actual and forecasted values in the GUI
        plt.figure(figsize=(10, 5))
        plt.plot(data.index[-(horizon + 1):], data['EnergyConsumption'].values[-(horizon + 1):], label='Actual', marker='o')
        plt.plot(forecast_index[1:], forecast, label='Forecast', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.title('Energy Consumption Forecast')
        plt.legend()
        plt.show()

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter a valid number of days to forecast.")

# Add a button to initiate forecasting
button = ttk.Button(root, text="Forecast", command=make_forecast)
button.pack()

root.mainloop()
