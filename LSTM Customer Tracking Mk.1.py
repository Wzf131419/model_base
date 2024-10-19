# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the outbound data
out_data = pd.read_csv('5-12 out.csv')

# Data Preprocessing
# Rename columns for easier processing
out_data.rename(columns=lambda x: x.strip(), inplace=True)
date_column_out = out_data.columns[2]
quantity_out_column = out_data.columns[20]
out_data.rename(columns={date_column_out: 'date', quantity_out_column: 'quantity_out'}, inplace=True)

# Drop rows with missing essential information
out_data.dropna(subset=['date', 'quantity_out'], inplace=True)
# Convert 'date' column to datetime
out_data['date'] = pd.to_datetime(out_data['date'], errors='coerce')
out_data.dropna(subset=['date'], inplace=True)

# Group by customer and prepare time series data for each customer
customer_column_out = out_data.columns[7]  # Assuming the eighth column is the customer ID
out_data.rename(columns={customer_column_out: 'customer_id'}, inplace=True)

# Function to create time series dataset
def create_dataset(data, time_step=5):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Placeholder for anomaly detection results
anomalies = []

# Loop through each customer and build a model
for customer_id, customer_data in out_data.groupby('customer_id'):
    if customer_data.empty:
        continue

    customer_data = customer_data.sort_values('date')
    quantities = customer_data[['quantity_out']].values

    # Scale the data
    scaled_data = scaler.fit_transform(quantities)

    # Create the dataset
    time_step = 5
    X, y = create_dataset(scaled_data, time_step)
    if len(X) == 0 or len(y) == 0:
        continue

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Skip training if there are not enough samples
    if len(X_train) == 0 or len(y_train) == 0:
        continue

    # Adjust batch size if larger than training set
    batch_size = min(32, len(X_train))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=10)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate residuals and detect anomalies
    residuals = np.abs(predictions - y_test_scaled)
    threshold = np.mean(residuals) + 2 * np.std(residuals)  # Set threshold for anomaly detection
    anomaly_indices = np.where(residuals > threshold)[0]

    # Store anomalies with customer ID and date
    for idx in anomaly_indices:
        anomalies.append({
            'customer_id': customer_id,
            'date': customer_data.iloc[train_size + time_step + idx]['date'],
            'actual_quantity': y_test_scaled[idx][0],
            'predicted_quantity': predictions[idx][0]
        })

# Convert anomalies to DataFrame and save to CSV
anomalies_df = pd.DataFrame(anomalies)
anomalies_df.to_csv('customer_anomalies_report.csv', index=False)

# Plot example for one customer
if len(y_test_scaled) > 0 and len(predictions) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_scaled, label='Actual Quantity')
    plt.plot(predictions, label='Predicted Quantity')
    plt.xlabel('Time')
    plt.ylabel('Quantity Out')
    plt.title('Actual vs Predicted Quantity for Customer')
    plt.legend()
    plt.show()


