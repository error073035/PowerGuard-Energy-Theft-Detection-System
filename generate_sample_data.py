#!/usr/bin/env python3
"""
Generate sample datasets for PowerGuard testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_normal_dataset(filename: str = "data/normal_power_consumption.csv", days: int = 30):
    """Generate a dataset with mostly normal power consumption patterns"""
    
    print(f"📊 Generating normal dataset: {filename}")
    
    # Create date range
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]  # Remove last to get exact hours
    
    data = []
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.weekday()
        
        # Base consumption pattern (typical residential/commercial)
        if 0 <= hour <= 5:  # Night - low consumption
            base_consumption = np.random.normal(1.5, 0.3)
        elif 6 <= hour <= 8:  # Morning peak
            base_consumption = np.random.normal(3.5, 0.5)
        elif 9 <= hour <= 16:  # Day - moderate consumption
            base_consumption = np.random.normal(2.5, 0.4)
        elif 17 <= hour <= 21:  # Evening peak
            base_consumption = np.random.normal(4.0, 0.6)
        else:  # Late evening
            base_consumption = np.random.normal(2.0, 0.3)
        
        # Weekend adjustment (slightly different pattern)
        if day_of_week >= 5:  # Weekend
            base_consumption *= np.random.uniform(0.8, 1.2)
        
        # Add some random variation
        consumption = max(0.1, base_consumption + np.random.normal(0, 0.2))
        
        data.append({
            'Datetime': dt,
            'PowerConsumption_Zone1': round(consumption, 2)
        })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Generated {len(df)} normal consumption records")
    return df

def generate_anomaly_dataset(filename: str = "data/power_with_anomalies.csv", days: int = 30):
    """Generate a dataset with clear anomalies (energy theft patterns)"""
    
    print(f"🚨 Generating anomaly dataset: {filename}")
    
    # Start with normal data
    df = generate_normal_dataset("temp_normal.csv", days)
    
    # Inject anomalies (approximately 5% of data)
    num_anomalies = int(len(df) * 0.05)
    anomaly_indices = np.random.choice(len(df), num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Create different types of anomalies
        anomaly_type = np.random.choice(['spike', 'sustained_high', 'unusual_pattern'])
        
        if anomaly_type == 'spike':
            # Sudden spike (10-20x normal consumption)
            df.loc[idx, 'PowerConsumption_Zone1'] *= np.random.uniform(10, 20)
        
        elif anomaly_type == 'sustained_high':
            # Sustained high consumption for several hours
            duration = np.random.randint(3, 8)
            end_idx = min(idx + duration, len(df))
            multiplier = np.random.uniform(5, 10)
            df.loc[idx:end_idx, 'PowerConsumption_Zone1'] *= multiplier
        
        elif anomaly_type == 'unusual_pattern':
            # Unusual consumption at odd hours (e.g., very high consumption at 3 AM)
            hour = df.loc[idx, 'Datetime'].hour
            if 0 <= hour <= 5:  # Night hours
                df.loc[idx, 'PowerConsumption_Zone1'] *= np.random.uniform(8, 15)
    
    # Round values
    df['PowerConsumption_Zone1'] = df['PowerConsumption_Zone1'].round(2)
    
    # Save dataset
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    
    # Clean up temp file
    if os.path.exists("temp_normal.csv"):
        os.remove("temp_normal.csv")
    
    print(f"✅ Generated {len(df)} records with {num_anomalies} anomalies")
    return df

def generate_corrupted_dataset(filename: str = "data/corrupted_power_data.csv", days: int = 20):
    """Generate a dataset with missing values and data quality issues"""
    
    print(f"🔧 Generating corrupted dataset: {filename}")
    
    # Start with normal data
    df = generate_normal_dataset("temp_normal.csv", days)
    
    # Introduce various data quality issues
    
    # 1. Missing values (random)
    missing_indices = np.random.choice(len(df), int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, 'PowerConsumption_Zone1'] = np.nan
    
    # 2. String values that should be numeric
    string_indices = np.random.choice(len(df), int(len(df) * 0.02), replace=False)
    df.loc[string_indices, 'PowerConsumption_Zone1'] = '?'
    
    # 3. Negative values (sensor errors)
    negative_indices = np.random.choice(len(df), int(len(df) * 0.01), replace=False)
    df.loc[negative_indices, 'PowerConsumption_Zone1'] = -np.random.uniform(0.1, 2.0)
    
    # 4. Extreme outliers (sensor malfunction)
    outlier_indices = np.random.choice(len(df), int(len(df) * 0.01), replace=False)
    df.loc[outlier_indices, 'PowerConsumption_Zone1'] = np.random.uniform(1000, 10000)
    
    # 5. Some missing timestamps (create gaps)
    gap_indices = np.random.choice(len(df), int(len(df) * 0.02), replace=False)
    df = df.drop(gap_indices).reset_index(drop=True)
    
    # 6. Add some NULL string values
    null_indices = np.random.choice(len(df), int(len(df) * 0.01), replace=False)
    df.loc[null_indices, 'PowerConsumption_Zone1'] = 'NULL'
    
    # Save dataset
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    
    # Clean up temp file
    if os.path.exists("temp_normal.csv"):
        os.remove("temp_normal.csv")
    
    print(f"✅ Generated {len(df)} records with various data quality issues")
    return df

def generate_different_format_dataset(filename: str = "data/alternative_format.csv", days: int = 25):
    """Generate dataset with different column names to test auto-detection"""
    
    print(f"🔄 Generating alternative format dataset: {filename}")
    
    # Create date range
    start_date = datetime(2024, 2, 1)
    end_date = start_date + timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]
    
    data = []
    
    for dt in date_range:
        hour = dt.hour
        
        # Different consumption pattern
        if 0 <= hour <= 6:
            base_consumption = np.random.normal(2.0, 0.4)
        elif 7 <= hour <= 9:
            base_consumption = np.random.normal(4.5, 0.7)
        elif 10 <= hour <= 15:
            base_consumption = np.random.normal(3.2, 0.5)
        elif 16 <= hour <= 20:
            base_consumption = np.random.normal(5.0, 0.8)
        else:
            base_consumption = np.random.normal(2.5, 0.4)
        
        consumption = max(0.1, base_consumption + np.random.normal(0, 0.3))
        
        # Add some anomalies (3% chance)
        if np.random.random() < 0.03:
            consumption *= np.random.uniform(8, 15)
        
        data.append({
            'timestamp': dt,  # Different column name
            'AEP_MW': round(consumption, 2)  # Different column name
        })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Generated {len(df)} records with alternative column names")
    return df

def main():
    """Generate all sample datasets"""
    print("🚀 Generating PowerGuard Sample Datasets")
    print("=" * 50)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate different types of datasets
    datasets = [
        ("Normal Power Consumption", generate_normal_dataset, "data/normal_power_consumption.csv", 30),
        ("Power with Anomalies", generate_anomaly_dataset, "data/power_with_anomalies.csv", 30),
        ("Corrupted Data", generate_corrupted_dataset, "data/corrupted_power_data.csv", 20),
        ("Alternative Format", generate_different_format_dataset, "data/alternative_format.csv", 25)
    ]
    
    for name, func, filename, days in datasets:
        print(f"\n📊 {name}")
        print("-" * 30)
        df = func(filename, days)
        print(f"📁 Saved to: {filename}")
        print(f"📈 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📊 Power consumption range: {df.iloc[:, 1].min():.2f} - {df.iloc[:, 1].max():.2f}")
    
    print("\n" + "=" * 50)
    print("✅ All sample datasets generated successfully!")
    print("\n🔧 Next steps:")
    print("1. Train a model: python train_model.py --data-path data/power_with_anomalies.csv")
    print("2. Run the app: streamlit run app.py")

if __name__ == "__main__":
    main()