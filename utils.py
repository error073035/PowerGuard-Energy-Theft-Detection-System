import pandas as pd
import numpy as np
import yaml
import logging
import os
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import joblib

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'powerguard.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")

def detect_columns(df: pd.DataFrame, config: dict, logger: logging.Logger) -> Tuple[str, str]:
    """
    Automatically detect timestamp and power consumption columns
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (timestamp_column, power_column)
    """
    timestamp_col = None
    power_col = None
    
    # Convert column names to lowercase for pattern matching
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Detect timestamp column
    timestamp_patterns = config['data']['timestamp_patterns']
    for pattern in timestamp_patterns:
        for col_lower, col_original in columns_lower.items():
            if pattern in col_lower:
                timestamp_col = col_original
                logger.info(f"✅ Detected timestamp column: {timestamp_col}")
                break
        if timestamp_col:
            break
    
    if not timestamp_col:
        # Try to find datetime-like columns by data type
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(10))
                    timestamp_col = col
                    logger.info(f"✅ Detected timestamp column by type: {timestamp_col}")
                    break
                except:
                    continue
    
    # Detect power consumption column
    power_patterns = config['data']['power_patterns']
    for pattern in power_patterns:
        for col_lower, col_original in columns_lower.items():
            if pattern in col_lower:
                power_col = col_original
                logger.info(f"✅ Detected power column: {power_col}")
                break
        if power_col:
            break
    
    if not power_col:
        # Find numeric columns that could be power consumption
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Choose the column with highest variance (likely power consumption)
            variances = df[numeric_cols].var()
            power_col = variances.idxmax()
            logger.info(f"✅ Detected power column by variance: {power_col}")
    
    if not timestamp_col:
        raise ValueError("❌ Could not detect timestamp column. Please ensure your dataset has a datetime column.")
    
    if not power_col:
        raise ValueError("❌ Could not detect power consumption column. Please ensure your dataset has a numeric power column.")
    
    return timestamp_col, power_col

def clean_data(df: pd.DataFrame, timestamp_col: str, power_col: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Clean and preprocess the dataset
    
    Args:
        df: Input dataframe
        timestamp_col: Name of timestamp column
        power_col: Name of power consumption column
        logger: Logger instance
        
    Returns:
        Cleaned dataframe
    """
    logger.info("🧹 Starting data cleaning...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle missing values in timestamp column
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[timestamp_col])
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with missing timestamps")
    
    # Convert timestamp column
    try:
        df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col])
        logger.info(f"✅ Successfully converted {timestamp_col} to datetime")
    except Exception as e:
        raise ValueError(f"❌ Could not convert {timestamp_col} to datetime: {e}")
    
    # Clean power consumption column
    # Replace common missing value indicators
    df_clean[power_col] = df_clean[power_col].replace(['?', 'NULL', 'null', 'NaN', ''], np.nan)
    
    # Convert to numeric
    df_clean[power_col] = pd.to_numeric(df_clean[power_col], errors='coerce')
    
    # Handle missing values in power column
    missing_power = df_clean[power_col].isna().sum()
    if missing_power > 0:
        logger.info(f"Found {missing_power} missing values in power column")
        
        # Forward fill then backward fill for time series data
        df_clean[power_col] = df_clean[power_col].fillna(method='ffill').fillna(method='bfill')
        
        # If still missing, use median
        if df_clean[power_col].isna().sum() > 0:
            median_value = df_clean[power_col].median()
            df_clean[power_col] = df_clean[power_col].fillna(median_value)
            logger.info(f"Filled remaining missing values with median: {median_value:.2f}")
    
    # Remove outliers (values beyond 5 standard deviations)
    mean_val = df_clean[power_col].mean()
    std_val = df_clean[power_col].std()
    outlier_threshold = 5 * std_val
    
    outliers = (df_clean[power_col] < (mean_val - outlier_threshold)) | \
               (df_clean[power_col] > (mean_val + outlier_threshold))
    
    outlier_count = outliers.sum()
    if outlier_count > 0:
        logger.info(f"Removing {outlier_count} extreme outliers")
        df_clean = df_clean[~outliers]
    
    # Sort by timestamp
    df_clean = df_clean.sort_values(timestamp_col).reset_index(drop=True)
    
    logger.info(f"✅ Data cleaning completed. Final dataset size: {len(df_clean)} rows")
    return df_clean

def engineer_features(df: pd.DataFrame, timestamp_col: str, power_col: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Create features for machine learning model
    
    Args:
        df: Input dataframe
        timestamp_col: Name of timestamp column
        power_col: Name of power consumption column
        logger: Logger instance
        
    Returns:
        Dataframe with engineered features
    """
    logger.info("⚡ Starting feature engineering...")
    
    df_features = df.copy()
    
    # Rename power column to standard name
    df_features['value'] = df_features[power_col]
    
    # Extract time-based features
    df_features['hour'] = df_features[timestamp_col].dt.hour
    df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek
    df_features['month'] = df_features[timestamp_col].dt.month
    df_features['day_of_year'] = df_features[timestamp_col].dt.dayofyear
    
    # Create lag features
    df_features['lag_1'] = df_features['value'].shift(1)
    df_features['lag_24'] = df_features['value'].shift(24)
    df_features['lag_168'] = df_features['value'].shift(168)  # Same hour last week
    
    # Rolling statistics
    df_features['rolling_mean_24'] = df_features['value'].rolling(window=24, min_periods=1).mean()
    df_features['rolling_std_24'] = df_features['value'].rolling(window=24, min_periods=1).std()
    
    # Ratio features
    df_features['value_to_lag1_ratio'] = df_features['value'] / (df_features['lag_1'] + 1e-8)
    df_features['value_to_rolling_mean_ratio'] = df_features['value'] / (df_features['rolling_mean_24'] + 1e-8)
    
    # Drop rows with NaN values created by lag features
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    logger.info(f"Removed {initial_rows - len(df_features)} rows due to lag feature creation")
    
    logger.info(f"✅ Feature engineering completed. Features created: {len(df_features.columns)}")
    return df_features

def create_anomaly_labels(df: pd.DataFrame, threshold_factor: float, logger: logging.Logger) -> pd.DataFrame:
    """
    Create anomaly labels based on statistical threshold
    
    Args:
        df: Input dataframe with 'value' column
        threshold_factor: Multiplier for standard deviation
        logger: Logger instance
        
    Returns:
        Dataframe with 'label' column added
    """
    logger.info("🚨 Creating anomaly labels...")
    
    df_labeled = df.copy()
    
    # Calculate threshold
    mean_val = df_labeled['value'].mean()
    std_val = df_labeled['value'].std()
    threshold = mean_val + threshold_factor * std_val
    
    # Create binary labels
    df_labeled['label'] = (df_labeled['value'] > threshold).astype(int)
    
    anomaly_count = df_labeled['label'].sum()
    anomaly_percentage = (anomaly_count / len(df_labeled)) * 100
    
    logger.info(f"📊 Anomaly threshold: {threshold:.2f}")
    logger.info(f"📊 Total anomalies detected: {anomaly_count} ({anomaly_percentage:.2f}%)")
    
    return df_labeled

def save_model_and_scaler(model, scaler: Optional[StandardScaler], model_dir: str, 
                         model_file: str, scaler_file: str, logger: logging.Logger):
    """Save trained model and scaler"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, model_file)
    joblib.dump(model, model_path)
    logger.info(f"💾 Model saved to: {model_path}")
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(model_dir, scaler_file)
        joblib.dump(scaler, scaler_path)
        logger.info(f"💾 Scaler saved to: {scaler_path}")

def load_model_and_scaler(model_dir: str, model_file: str, scaler_file: str, 
                         logger: logging.Logger) -> Tuple:
    """Load trained model and scaler"""
    model_path = os.path.join(model_dir, model_file)
    scaler_path = os.path.join(model_dir, scaler_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"📂 Model loaded from: {model_path}")
    
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info(f"📂 Scaler loaded from: {scaler_path}")
    
    return model, scaler

def get_feature_columns() -> List[str]:
    """Get the list of feature columns used for training"""
    return [
        'hour', 'day_of_week', 'month', 'day_of_year',
        'lag_1', 'lag_24', 'lag_168',
        'rolling_mean_24', 'rolling_std_24',
        'value_to_lag1_ratio', 'value_to_rolling_mean_ratio',
        'value'
    ]