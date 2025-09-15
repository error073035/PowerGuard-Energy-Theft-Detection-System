#!/usr/bin/env python3
"""
PowerGuard Energy Theft Detection - Training Pipeline
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

from utils import (
    setup_logging, load_config, detect_columns, clean_data,
    engineer_features, create_anomaly_labels, save_model_and_scaler,
    get_feature_columns
)

def train_model(data_path: str, config_path: str = "config.yaml"):
    """
    Main training pipeline
    
    Args:
        data_path: Path to the training dataset
        config_path: Path to configuration file
    """
    # Setup
    config = load_config(config_path)
    logger = setup_logging(config['paths']['logs_dir'])
    
    logger.info("🚀 Starting PowerGuard Training Pipeline")
    logger.info(f"📁 Dataset: {data_path}")
    logger.info(f"⚙️ Config: {config_path}")
    
    try:
        # Step 1: Load data
        logger.info("📊 Step 1: Loading dataset...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        logger.info(f"📋 Columns: {list(df.columns)}")
        
        # Step 2: Detect columns automatically
        logger.info("🔍 Step 2: Detecting columns...")
        timestamp_col, power_col = detect_columns(df, config, logger)
        
        # Step 3: Clean data
        logger.info("🧹 Step 3: Cleaning data...")
        df_clean = clean_data(df, timestamp_col, power_col, logger)
        
        # Check minimum samples requirement
        min_samples = config['anomaly_detection']['min_samples_for_training']
        if len(df_clean) < min_samples:
            raise ValueError(f"❌ Insufficient data. Need at least {min_samples} samples, got {len(df_clean)}")
        
        # Step 4: Feature engineering
        logger.info("⚡ Step 4: Engineering features...")
        df_features = engineer_features(df_clean, timestamp_col, power_col, logger)
        
        # Step 5: Create anomaly labels
        logger.info("🚨 Step 5: Creating anomaly labels...")
        threshold_factor = config['anomaly_detection']['threshold_factor']
        df_labeled = create_anomaly_labels(df_features, threshold_factor, logger)
        
        # Step 6: Prepare training data
        logger.info("📊 Step 6: Preparing training data...")
        feature_columns = get_feature_columns()
        
        # Ensure all feature columns exist
        missing_features = [col for col in feature_columns if col not in df_labeled.columns]
        if missing_features:
            raise ValueError(f"❌ Missing feature columns: {missing_features}")
        
        X = df_labeled[feature_columns]
        y = df_labeled['label']
        
        logger.info(f"📈 Feature matrix shape: {X.shape}")
        logger.info(f"🎯 Target distribution: Normal={sum(y==0)}, Anomaly={sum(y==1)}")
        
        # Step 7: Split data
        logger.info("✂️ Step 7: Splitting data...")
        test_size = config['model']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config['model']['random_state'], 
            stratify=y if y.sum() > 0 else None
        )
        
        logger.info(f"🏋️ Training set: {X_train.shape[0]} samples")
        logger.info(f"🧪 Test set: {X_test.shape[0]} samples")
        
        # Step 8: Handle class imbalance
        logger.info("⚖️ Step 8: Handling class imbalance...")
        if y_train.sum() > 0 and y_train.sum() < len(y_train):
            try:
                smote = SMOTE(random_state=config['model']['random_state'])
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"✅ SMOTE applied. New training size: {X_train_balanced.shape[0]}")
                X_train, y_train = X_train_balanced, y_train_balanced
            except Exception as e:
                logger.warning(f"⚠️ SMOTE failed: {e}. Proceeding with original data.")
        
        # Step 9: Scale features
        logger.info("📏 Step 9: Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 10: Train model
        logger.info("🤖 Step 10: Training model...")
        model = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state'],
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        logger.info("✅ Model training completed!")
        
        # Step 11: Evaluate model
        logger.info("📈 Step 11: Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"🎯 Accuracy: {accuracy:.4f}")
        
        # Detailed evaluation
        logger.info("\n📊 Classification Report:")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        logger.info("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{cm}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n🔍 Top 10 Feature Importances:")
        logger.info(f"\n{feature_importance.head(10).to_string(index=False)}")
        
        # Step 12: Save model and scaler
        logger.info("💾 Step 12: Saving model and scaler...")
        save_model_and_scaler(
            model, scaler,
            config['paths']['model_dir'],
            config['paths']['model_file'],
            config['paths']['scaler_file'],
            logger
        )
        
        # Save training metadata
        metadata = {
            'timestamp_column': timestamp_col,
            'power_column': power_col,
            'feature_columns': feature_columns,
            'threshold_factor': threshold_factor,
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'anomaly_percentage': (y.sum() / len(y)) * 100
        }
        
        metadata_path = os.path.join(config['paths']['model_dir'], 'training_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📋 Training metadata saved to: {metadata_path}")
        logger.info("🎉 Training pipeline completed successfully!")
        
        return model, scaler, accuracy
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="PowerGuard Energy Theft Detection - Training Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to training dataset CSV file")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--threshold-factor", type=float, help="Override threshold factor for anomaly detection")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.threshold_factor:
        config = load_config(args.config)
        config['anomaly_detection']['threshold_factor'] = args.threshold_factor
        
        # Save temporary config
        import yaml
        temp_config_path = "temp_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = temp_config_path
    else:
        config_path = args.config
    
    try:
        train_model(args.data_path, config_path)
        print("\n✅ Training completed successfully!")
        print("🚀 You can now run the Streamlit app: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    
    finally:
        # Clean up temporary config
        if args.threshold_factor and os.path.exists("temp_config.yaml"):
            os.remove("temp_config.yaml")

if __name__ == "__main__":
    main()