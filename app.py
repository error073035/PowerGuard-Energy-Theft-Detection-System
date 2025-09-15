import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model/energy_theft_detection_model.pkl')

st.title('⚡ PowerGuard – Energy Theft Detection System')

hour = st.slider('Hour (0-23)', 0, 23, 12)
day = st.slider('Day of Week (0=Monday, 6=Sunday)', 0, 6, 1)
month = st.slider('Month (1-12)', 1, 12, 8)
lag1 = st.number_input('Previous Hour Power Consumption (kWh)', 0.0, 10.0, 1.0)
lag24 = st.number_input('Power Consumption Same Hour Yesterday (kWh)', 0.0, 10.0, 1.0)
current_power = st.number_input('Current Power Consumption (kWh)', 0.0, 20.0, 1.5)

if st.button('Check Anomaly'):
    input_df = pd.DataFrame([[hour, day, month, lag1, lag24, current_power]],
                            columns=['hour', 'day_of_week', 'month', 'lag_1', 'lag_24', 'value'])

    st.write("📝 Input Data Preview:")
    st.dataframe(input_df)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error('⚠️ Anomaly Detected: Possible Energy Theft!')
    else:
        st.success('✔️ Usage is Normal')
