# ⚡ PowerGuard – Energy Theft Detection System

A robust, dataset-agnostic machine learning system for detecting anomalies in power consumption data that could indicate energy theft or unusual usage patterns.

## 🌟 Features

- **🤖 Automatic Dataset Adaptation**: Automatically detects timestamp and power consumption columns regardless of naming conventions
- **🧹 Intelligent Data Cleaning**: Handles missing values, corrupted data, and outliers automatically
- **⚡ Advanced Feature Engineering**: Creates lag features, rolling statistics, and ratio-based features
- **🎯 Configurable Anomaly Detection**: Adjustable threshold-based anomaly labeling
- **📊 Interactive Web Interface**: User-friendly Streamlit app for real-time predictions
- **📈 Comprehensive Evaluation**: Detailed model performance metrics and feature importance analysis
- **🔧 Modular Architecture**: Clean, maintainable code structure with proper error handling

## 🏗️ Project Structure

```
PowerGuard/
├── config.yaml                 # Configuration file
├── utils.py                   # Utility functions
├── train_model.py            # Training pipeline
├── app.py                    # Streamlit web application
├── generate_sample_data.py   # Sample dataset generator
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
│   ├── normal_power_consumption.csv
│   ├── power_with_anomalies.csv
│   ├── corrupted_power_data.csv
│   └── alternative_format.csv
├── model/                    # Trained model directory
│   ├── energy_theft_detection_model.pkl
│   ├── feature_scaler.pkl
│   └── training_metadata.json
└── logs/                     # Log files
    └── powerguard.log
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd PowerGuard

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data (Optional)

```bash
# Generate sample datasets for testing
python generate_sample_data.py
```

### 3. Train the Model

```bash
# Train with sample data
python train_model.py --data-path data/power_with_anomalies.csv

# Or train with your own dataset
python train_model.py --data-path path/to/your/dataset.csv

# With custom threshold factor
python train_model.py --data-path your_data.csv --threshold-factor 2.5
```

### 4. Run the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 Dataset Requirements

Your dataset should contain:

1. **Timestamp Column**: Any column with datetime information
   - Supported names: `datetime`, `timestamp`, `settlement_date`, `date`, `time`
   - Format: Any format parseable by `pd.to_datetime()`

2. **Power Consumption Column**: Numeric column with power consumption values
   - Supported names: `aep_mw`, `powerconsumption`, `power_consumption`, `power`, `consumption`, `load`, `demand`
   - Units: Any consistent unit (kWh, MW, etc.)

### Example Dataset Formats

**Format 1:**
```csv
Datetime,PowerConsumption_Zone1
2024-01-01 00:00:00,2.5
2024-01-01 01:00:00,2.1
```

**Format 2:**
```csv
timestamp,AEP_MW
2024-01-01 00:00:00,150.2
2024-01-01 01:00:00,145.8
```

**Format 3:**
```csv
settlement_date,Power_Consumption
2024-01-01 00:00:00,3.2
2024-01-01 01:00:00,2.9
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  n_estimators: 100          # Random Forest trees
  test_size: 0.2            # Train/test split ratio

anomaly_detection:
  threshold_factor: 2.0      # Std deviation multiplier for anomaly threshold
  min_samples_for_training: 100

data:
  timestamp_patterns:        # Column name patterns for auto-detection
    - "datetime"
    - "timestamp"
  power_patterns:
    - "power"
    - "consumption"
```

## 🔧 Command Line Options

### Training Pipeline

```bash
python train_model.py --help

Options:
  --data-path PATH          Path to training dataset CSV file (required)
  --config PATH            Path to configuration file (default: config.yaml)
  --threshold-factor FLOAT Override threshold factor for anomaly detection
```

### Examples

```bash
# Basic training
python train_model.py --data-path data/my_power_data.csv

# Custom threshold (more sensitive to anomalies)
python train_model.py --data-path data/my_data.csv --threshold-factor 1.5

# Custom configuration
python train_model.py --data-path data/my_data.csv --config my_config.yaml
```

## 🎯 Using the Web Application

1. **Launch the app**: `streamlit run app.py`
2. **Enter power consumption data**:
   - Hour (0-23)
   - Day of week (Monday=0, Sunday=6)
   - Month (1-12)
   - Previous hour consumption
   - Same hour yesterday consumption
   - Current power consumption
3. **Click "Detect Anomaly"** to get prediction
4. **Review results** with confidence scores and contextual analysis

## 📈 Model Features

The system automatically creates these features for training:

- **Time-based**: hour, day_of_week, month, day_of_year
- **Lag features**: lag_1 (previous hour), lag_24 (same hour yesterday), lag_168 (same hour last week)
- **Rolling statistics**: 24-hour rolling mean and standard deviation
- **Ratio features**: current vs previous hour, current vs rolling average
- **Raw value**: current power consumption

## 🧪 Sample Datasets

The system includes 4 sample datasets for testing:

1. **normal_power_consumption.csv**: Clean data with typical consumption patterns
2. **power_with_anomalies.csv**: Data with injected anomalies (~5% anomaly rate)
3. **corrupted_power_data.csv**: Data with missing values and quality issues
4. **alternative_format.csv**: Different column names to test auto-detection

## 📊 Model Evaluation

After training, the system provides:

- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score for each class
- **Confusion Matrix**: True/false positives and negatives
- **Feature Importance**: Most important features for anomaly detection
- **Training Metadata**: Saved for reference in the web app

## 🔍 Troubleshooting

### Common Issues

1. **"Could not detect timestamp column"**
   - Ensure your dataset has a datetime column
   - Check column names match supported patterns
   - Verify datetime format is parseable

2. **"Could not detect power consumption column"**
   - Ensure you have a numeric power consumption column
   - Check column names match supported patterns
   - Verify data is numeric (not strings)

3. **"Insufficient data for training"**
   - Ensure dataset has at least 100 samples
   - Check for excessive missing values
   - Verify data cleaning didn't remove too many rows

4. **"Model file not found"**
   - Run the training pipeline first
   - Check the model directory exists
   - Verify training completed successfully

### Data Quality Tips

- **Remove extreme outliers** before training (values beyond 5 standard deviations)
- **Ensure consistent time intervals** (hourly data works best)
- **Handle missing values** appropriately for your use case
- **Validate timestamp formats** are consistent

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Scikit-learn](https://scikit-learn.org/) for machine learning
- [Streamlit](https://streamlit.io/) for the web interface
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance

---

**⚡ PowerGuard** - Protecting energy infrastructure through intelligent anomaly detection.