import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

print('🔧 Step 1: Loading Data...')
data = pd.read_csv('data/AEP_hourly.csv')
print(f'✅ Data Loaded Successfully! Columns available: {list(data.columns)}')

print('🧹 Step 2: Cleaning Data...')
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Use the correct column
data['value'] = data['AEP_MW'].astype(float)
print('✅ Using column: AEP_MW as value.')

print('🔧 Processing timestamp column...')
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Extract time-based features
data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['month'] = data['Datetime'].dt.month

print('⚡ Step 3: Feature Engineering...')
data['lag_1'] = data['value'].shift(1)
data['lag_24'] = data['value'].shift(24)
data.dropna(inplace=True)

print('🚨 Step 4: Labeling Anomalies...')
# Set threshold: mean + 3*std
threshold = data['value'].mean() + 3 * data['value'].std()
data['label'] = (data['value'] > threshold).astype(int)
print(f'📊 Total Anomalies Detected: {data["label"].sum()}')

print('📊 Step 5: Preparing Input & Output...')
X = data[['hour', 'day_of_week', 'month', 'lag_1', 'lag_24', 'value']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(f'⚙️ Training model on {len(X_train)} samples...')

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print('✅ Model training completed!')

print('📈 Step 6: Evaluating Model...')
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print('💾 Step 7: Saving Model...')
joblib.dump(model, 'model/energy_theft_detection_model.pkl')
print('✅ Model saved successfully at model/energy_theft_detection_model.pkl')
