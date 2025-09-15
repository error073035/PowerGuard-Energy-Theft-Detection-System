import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from typing import Optional, Tuple

from utils import load_config, setup_logging, load_model_and_scaler, get_feature_columns

# Page configuration
st.set_page_config(
    page_title="PowerGuard - Energy Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_cached():
    """Load model and scaler with caching"""
    try:
        config = load_config()
        logger = setup_logging(config['paths']['logs_dir'])
        
        model, scaler = load_model_and_scaler(
            config['paths']['model_dir'],
            config['paths']['model_file'],
            config['paths']['scaler_file'],
            logger
        )
        
        # Load training metadata if available
        metadata_path = os.path.join(config['paths']['model_dir'], 'training_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, scaler, metadata, None
        
    except Exception as e:
        return None, None, {}, str(e)

def validate_inputs(hour: int, day: int, month: int, lag1: float, lag24: float, current: float) -> Optional[str]:
    """Validate user inputs"""
    if not (0 <= hour <= 23):
        return "Hour must be between 0 and 23"
    
    if not (0 <= day <= 6):
        return "Day of week must be between 0 (Monday) and 6 (Sunday)"
    
    if not (1 <= month <= 12):
        return "Month must be between 1 and 12"
    
    if lag1 < 0:
        return "Previous hour consumption cannot be negative"
    
    if lag24 < 0:
        return "Same hour yesterday consumption cannot be negative"
    
    if current < 0:
        return "Current power consumption cannot be negative"
    
    return None

def create_feature_vector(hour: int, day: int, month: int, lag1: float, lag24: float, 
                         current: float, day_of_year: int = 1) -> pd.DataFrame:
    """Create feature vector for prediction"""
    
    # Calculate additional features
    lag168 = lag24  # Approximate same hour last week with yesterday's value
    rolling_mean_24 = (lag1 + lag24) / 2  # Simple approximation
    rolling_std_24 = abs(lag1 - lag24) / 2  # Simple approximation
    
    # Ratio features (add small epsilon to avoid division by zero)
    value_to_lag1_ratio = current / (lag1 + 1e-8)
    value_to_rolling_mean_ratio = current / (rolling_mean_24 + 1e-8)
    
    # Create feature vector matching training format
    features = {
        'hour': hour,
        'day_of_week': day,
        'month': month,
        'day_of_year': day_of_year,
        'lag_1': lag1,
        'lag_24': lag24,
        'lag_168': lag168,
        'rolling_mean_24': rolling_mean_24,
        'rolling_std_24': rolling_std_24,
        'value_to_lag1_ratio': value_to_lag1_ratio,
        'value_to_rolling_mean_ratio': value_to_rolling_mean_ratio,
        'value': current
    }
    
    return pd.DataFrame([features])

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("⚡ PowerGuard – Energy Theft Detection System")
    st.markdown("---")
    
    # Load model
    model, scaler, metadata, error = load_model_cached()
    
    if error:
        st.error(f"❌ **Model Loading Error**: {error}")
        st.info("🔧 **Solution**: Please run the training pipeline first:")
        st.code("python train_model.py --data-path your_dataset.csv")
        st.stop()
    
    if model is None:
        st.error("❌ **No trained model found**")
        st.info("🔧 **Solution**: Please run the training pipeline first:")
        st.code("python train_model.py --data-path your_dataset.csv")
        st.stop()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        if metadata:
            st.metric("Model Accuracy", f"{metadata.get('accuracy', 0):.1%}")
            st.metric("Training Samples", f"{metadata.get('training_samples', 0):,}")
            st.metric("Anomaly Rate", f"{metadata.get('anomaly_percentage', 0):.2f}%")
            
            with st.expander("📋 Training Details"):
                st.write(f"**Timestamp Column**: {metadata.get('timestamp_column', 'N/A')}")
                st.write(f"**Power Column**: {metadata.get('power_column', 'N/A')}")
                st.write(f"**Threshold Factor**: {metadata.get('threshold_factor', 'N/A')}")
        else:
            st.info("Model metadata not available")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Enter the required power consumption data**
        2. **Click 'Detect Anomaly'** to get prediction
        3. **Review the result** and confidence score
        """)
    
    # Main input form
    st.header("🔍 Anomaly Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Time Information")
        hour = st.slider("Hour (0-23)", 0, 23, 12, help="Hour of the day (0 = midnight, 12 = noon)")
        day = st.selectbox(
            "Day of Week", 
            options=list(range(7)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=1,
            help="Day of the week"
        )
        month = st.slider("Month (1-12)", 1, 12, 8, help="Month of the year")
    
    with col2:
        st.subheader("⚡ Power Consumption Data")
        lag1 = st.number_input(
            "Previous Hour Consumption (kWh)", 
            min_value=0.0, 
            max_value=100.0, 
            value=1.0, 
            step=0.1,
            help="Power consumption in the previous hour"
        )
        lag24 = st.number_input(
            "Same Hour Yesterday (kWh)", 
            min_value=0.0, 
            max_value=100.0, 
            value=1.0, 
            step=0.1,
            help="Power consumption at the same hour yesterday"
        )
        current_power = st.number_input(
            "Current Power Consumption (kWh)", 
            min_value=0.0, 
            max_value=100.0, 
            value=1.5, 
            step=0.1,
            help="Current power consumption reading"
        )
    
    # Prediction section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Detect Anomaly", type="primary", use_container_width=True):
            
            # Validate inputs
            validation_error = validate_inputs(hour, day, month, lag1, lag24, current_power)
            if validation_error:
                st.error(f"❌ **Input Error**: {validation_error}")
                st.stop()
            
            try:
                # Create feature vector
                day_of_year = 1  # Default value since we don't have full date
                input_df = create_feature_vector(hour, day, month, lag1, lag24, current_power, day_of_year)
                
                # Display input data preview
                st.subheader("📝 Input Data Preview")
                display_df = pd.DataFrame({
                    'Hour': [hour],
                    'Day of Week': [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day]],
                    'Month': [month],
                    'Previous Hour (kWh)': [lag1],
                    'Same Hour Yesterday (kWh)': [lag24],
                    'Current Consumption (kWh)': [current_power]
                })
                st.dataframe(display_df, use_container_width=True)
                
                # Make prediction
                feature_columns = get_feature_columns()
                X = input_df[feature_columns]
                
                # Scale features if scaler is available
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X
                
                # Get prediction and probability
                prediction = model.predict(X_scaled)[0]
                prediction_proba = model.predict_proba(X_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Detection Result")
                
                if prediction == 1:
                    st.error("⚠️ **ANOMALY DETECTED: Possible Energy Theft!**")
                    confidence = prediction_proba[1] * 100
                    st.error(f"🔴 **Confidence**: {confidence:.1f}%")
                    
                    # Additional context
                    st.warning("**Recommended Actions:**")
                    st.warning("• Investigate this power consumption pattern")
                    st.warning("• Check for unauthorized connections")
                    st.warning("• Verify meter readings")
                    st.warning("• Consider on-site inspection")
                    
                else:
                    st.success("✅ **NORMAL USAGE DETECTED**")
                    confidence = prediction_proba[0] * 100
                    st.success(f"🟢 **Confidence**: {confidence:.1f}%")
                    
                    st.info("Power consumption appears to be within normal parameters.")
                
                # Show probability breakdown
                with st.expander("📊 Detailed Probability Breakdown"):
                    prob_df = pd.DataFrame({
                        'Classification': ['Normal Usage', 'Anomaly/Theft'],
                        'Probability': [f"{prediction_proba[0]:.1%}", f"{prediction_proba[1]:.1%}"]
                    })
                    st.dataframe(prob_df, use_container_width=True)
                
                # Contextual analysis
                st.markdown("---")
                st.subheader("📈 Contextual Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ratio_to_prev = current_power / (lag1 + 1e-8)
                    st.metric("vs Previous Hour", f"{ratio_to_prev:.2f}x")
                    if ratio_to_prev > 3:
                        st.warning("⚠️ Significant increase")
                    elif ratio_to_prev < 0.3:
                        st.info("📉 Notable decrease")
                
                with col2:
                    ratio_to_yesterday = current_power / (lag24 + 1e-8)
                    st.metric("vs Same Hour Yesterday", f"{ratio_to_yesterday:.2f}x")
                    if ratio_to_yesterday > 3:
                        st.warning("⚠️ Much higher than usual")
                    elif ratio_to_yesterday < 0.3:
                        st.info("📉 Much lower than usual")
                
                with col3:
                    avg_consumption = (lag1 + lag24) / 2
                    st.metric("Average Reference", f"{avg_consumption:.2f} kWh")
                
            except Exception as e:
                st.error(f"❌ **Prediction Error**: {str(e)}")
                st.error("Please check your inputs and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ PowerGuard Energy Theft Detection System</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()