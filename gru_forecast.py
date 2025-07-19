
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.colab import files

# 📁 Upload training data file (April to July)
print("📤 Select the 3-month weather data file")
uploaded_train = files.upload()
train_file = list(uploaded_train.keys())[0]

# Prepare training data
df = pd.read_csv(train_file, skiprows=13, names=['datetime', 'temperature'])
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%dT%H%M')
df.set_index('datetime', inplace=True)
df = df['2025-04-01':'2025-07-01 22:00']

# Normalize temperature values
scaler = MinMaxScaler()
df['temp_scaled'] = scaler.fit_transform(df[['temperature']])

# ⏳ Create sequences: input = past 72 hours, output = next 24 hours
n_steps_in = 72
n_steps_out = 24

def create_multi_sequences(data, n_in, n_out):
    X, y = [], []
    for i in range(n_in, len(data)-n_out):
        X.append(data[i-n_in:i])
        y.append(data[i:i+n_out])
    return np.array(X), np.array(y)

X, y = create_multi_sequences(df['temp_scaled'].values, n_steps_in, n_steps_out)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for GRU input

# 🧠 Build the GRU model
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(n_steps_in, 1)),
    Dropout(0.3),
    GRU(64),
    Dense(n_steps_out)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X, y, epochs=50, callbacks=[early_stop], verbose=1)

# 🔮 Forecast temperatures for July 2–9
last_input = df['temp_scaled'].values[-n_steps_in:]
predicted_scaled = []

for _ in range(7):  # Forecast 7 days
    x_input = last_input[-n_steps_in:].reshape((1, n_steps_in, 1))
    preds = model.predict(x_input, verbose=0)[0]
    predicted_scaled.extend(preds)
    last_input = np.append(last_input, preds)

# Convert predictions back to real temperature values
predicted = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()
forecast_dates = pd.date_range(start='2025-07-02', periods=7*24, freq='H')
forecast_df = pd.DataFrame({'datetime': forecast_dates, 'predicted_temp': predicted})
forecast_df.set_index('datetime', inplace=True)

# 📁 Upload actual temperature file for comparison
print("📤 Select the actual temperature file for July 2–9")
uploaded_true = files.upload()
true_file = list(uploaded_true.keys())[0]

# Prepare actual data
true_df = pd.read_csv(true_file, skiprows=13, names=['datetime', 'actual_temp'])
true_df['datetime'] = pd.to_datetime(true_df['datetime'], format='%Y%m%dT%H%M')
true_df.set_index('datetime', inplace=True)
true_df = true_df['2025-07-02':'2025-07-09 23:00']

# Merge predicted and actual data
compare_df = forecast_df.copy()
compare_df['actual_temp'] = true_df['actual_temp']
compare_df.dropna(inplace=True)

# 📊 Calculate evaluation metrics
mae = mean_absolute_error(compare_df['actual_temp'], compare_df['predicted_temp'])
rmse = np.sqrt(mean_squared_error(compare_df['actual_temp'], compare_df['predicted_temp']))

print(f"\n✅ MAE (Mean Absolute Error): {mae:.2f}°C")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:.2f}°C")

# 📉 Plot comparison between actual and predicted temperatures
plt.figure(figsize=(15,5))
plt.plot(compare_df.index, compare_df['actual_temp'], label='Actual Temperature', color='green')
plt.plot(compare_df.index, compare_df['predicted_temp'], label='Predicted Temperature', color='orange')
plt.title("📊 Actual vs Predicted Temperatures (July 2–9, 2025)")
plt.xlabel("Date & Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.show()
