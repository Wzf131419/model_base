# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the inbound and outbound data
in_data = pd.read_csv('5-12 in.csv')
out_data = pd.read_csv('5-12 out.csv')

# Data Preprocessing for Inbound Data
# Rename columns to English for easier processing
in_data.rename(columns=lambda x: x.strip(), inplace=True)
date_column_in = in_data.columns[2]
in_data.rename(columns={date_column_in: 'date'}, inplace=True)
settlement_unit_column_in = in_data.columns[5]  # Assuming the sixth column is the settlement unit ID
customer_name_column_in = in_data.columns[6]  # Assuming the seventh column is the customer name
quantity_in_column = in_data.columns[18]  # Assuming the nineteenth column is the quantity in
in_data.rename(columns={settlement_unit_column_in: 'settlement_unit_id', customer_name_column_in: 'customer_name', quantity_in_column: 'quantity_in'}, inplace=True)

# Drop rows with missing essential information
in_data.dropna(subset=['date', 'settlement_unit_id', 'quantity_in'], inplace=True)
# Convert 'date' columns to datetime for consistency
in_data['date'] = pd.to_datetime(in_data['date'], errors='coerce')
in_data.dropna(subset=['date'], inplace=True)

# Data Preprocessing for Outbound Data
out_data.rename(columns=lambda x: x.strip(), inplace=True)
date_column_out = out_data.columns[2]
out_data.rename(columns={date_column_out: 'date'}, inplace=True)
settlement_unit_column_out = out_data.columns[7]  # Assuming the eighth column is the settlement unit ID
customer_name_column_out = out_data.columns[6]  # Assuming the seventh column is the customer name
quantity_out_column = out_data.columns[20]  # Assuming the twenty-first column is the quantity out
out_data.rename(columns={settlement_unit_column_out: 'settlement_unit_id', customer_name_column_out: 'customer_name', quantity_out_column: 'quantity_out'}, inplace=True)

# Drop rows with missing essential information
out_data.dropna(subset=['date', 'settlement_unit_id', 'quantity_out'], inplace=True)
# Convert 'date' columns to datetime for consistency
out_data['date'] = pd.to_datetime(out_data['date'], errors='coerce')
out_data.dropna(subset=['date'], inplace=True)

# Modeling for Inbound Data
# Feature selection and scaling for inbound data
scaler_in = MinMaxScaler()
scaled_in_data = scaler_in.fit_transform(in_data[['quantity_in']])

# LSTM Model for Inbound Data
scaled_in_data = np.expand_dims(scaled_in_data, axis=0)  # Reshape to (1, timesteps, features)
model_in = Sequential()
model_in.add(LSTM(50, return_sequences=True, input_shape=(scaled_in_data.shape[1], scaled_in_data.shape[2])))
model_in.add(Dropout(0.2))
model_in.add(LSTM(50, return_sequences=False))
model_in.add(Dropout(0.2))
model_in.add(Dense(1))
model_in.compile(optimizer='adam', loss='mean_squared_error')

# Prepare data for training inbound model
X_train_in = []
y_train_in = []
for i in range(20, scaled_in_data.shape[1]):
    X_train_in.append(scaled_in_data[0, i-20:i, :])
    y_train_in.append(scaled_in_data[0, i, 0])
X_train_in, y_train_in = np.array(X_train_in), np.array(y_train_in)

# Train the LSTM model for inbound data
model_in.fit(X_train_in, y_train_in, epochs=10, batch_size=32)

# Predict future values for inbound data
predictions_in = model_in.predict(X_train_in)

# Adjust length of in_data to match predictions length
adjusted_in_data = in_data.iloc[20:].reset_index(drop=True)
customer_stickiness_in = adjusted_in_data[['date', 'settlement_unit_id', 'customer_name']].copy()
customer_stickiness_in['predicted_quantity'] = predictions_in.flatten()

# Calculating stickiness score for inbound data (aggregated by settlement_unit_id)
stickiness_score_in = customer_stickiness_in.groupby(['settlement_unit_id', 'customer_name'])['predicted_quantity'].sum().reset_index()
stickiness_score_in['stickiness_score'] = (stickiness_score_in['predicted_quantity'] / stickiness_score_in['predicted_quantity'].max()) * 10

# Save the inbound stickiness report to a CSV file
stickiness_score_in.to_csv('customer_stickiness_report_in.csv', index=False)

# Modeling for Outbound Data
# Feature selection and scaling for outbound data
scaler_out = MinMaxScaler()
scaled_out_data = scaler_out.fit_transform(out_data[['quantity_out']])

# LSTM Model for Outbound Data
scaled_out_data = np.expand_dims(scaled_out_data, axis=0)  # Reshape to (1, timesteps, features)
model_out = Sequential()
model_out.add(LSTM(50, return_sequences=True, input_shape=(scaled_out_data.shape[1], scaled_out_data.shape[2])))
model_out.add(Dropout(0.2))
model_out.add(LSTM(50, return_sequences=False))
model_out.add(Dropout(0.2))
model_out.add(Dense(1))
model_out.compile(optimizer='adam', loss='mean_squared_error')

# Prepare data for training outbound model
X_train_out = []
y_train_out = []
for i in range(20, scaled_out_data.shape[1]):
    X_train_out.append(scaled_out_data[0, i-20:i, :])
    y_train_out.append(scaled_out_data[0, i, 0])
X_train_out, y_train_out = np.array(X_train_out), np.array(y_train_out)

# Train the LSTM model for outbound data
model_out.fit(X_train_out, y_train_out, epochs=10, batch_size=32)

# Predict future values for outbound data
predictions_out = model_out.predict(X_train_out)

# Adjust length of out_data to match predictions length
adjusted_out_data = out_data.iloc[20:].reset_index(drop=True)
customer_stickiness_out = adjusted_out_data[['date', 'settlement_unit_id', 'customer_name']].copy()
customer_stickiness_out['predicted_quantity'] = predictions_out.flatten()

# Calculating stickiness score for outbound data (aggregated by settlement_unit_id)
stickiness_score_out = customer_stickiness_out.groupby(['settlement_unit_id', 'customer_name'])['predicted_quantity'].sum().reset_index()
stickiness_score_out['stickiness_score'] = (stickiness_score_out['predicted_quantity'] / stickiness_score_out['predicted_quantity'].max()) * 10

# Save the outbound stickiness report to a CSV file
stickiness_score_out.to_csv('customer_stickiness_report_out.csv', index=False)

# Visualization
# Load the stickiness score reports
stickiness_score_in = pd.read_csv('customer_stickiness_report_in.csv')
stickiness_score_out = pd.read_csv('customer_stickiness_report_out.csv')

import matplotlib.font_manager as fm

# Set up the font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# Plot top 10 inbound customers by stickiness score
top_10_customers_in = stickiness_score_in.sort_values(by='stickiness_score', ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10_customers_in['customer_name'], top_10_customers_in['stickiness_score'], color='blue', alpha=0.7, label='Stickiness Score')
plt.plot(top_10_customers_in['customer_name'], top_10_customers_in['stickiness_score'], color='red', marker='o', linestyle='-', linewidth=2, label='Trend Line')
plt.xlabel('Customer Name (Inbound)')
plt.ylabel('Stickiness Score (Scale 1-10)')
plt.title('Top 10 Inbound Customers by Stickiness Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot top 10 outbound customers by stickiness score
top_10_customers_out = stickiness_score_out.sort_values(by='stickiness_score', ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10_customers_out['customer_name'], top_10_customers_out['stickiness_score'], color='green', alpha=0.7, label='Stickiness Score')
plt.plot(top_10_customers_out['customer_name'], top_10_customers_out['stickiness_score'], color='red', marker='o', linestyle='-', linewidth=2, label='Trend Line')
plt.xlabel('Customer Name (Outbound)')
plt.ylabel('Stickiness Score (Scale 1-10)')
plt.title('Top 10 Outbound Customers by Stickiness Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
