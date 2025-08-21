# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load the trained model
reg = joblib.load('aqi_model.joblib')

# Load the data for visualization and default values
@st.cache_data
def load_data():
    df = pd.read_csv('delhi_aqi.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

df = load_data()

# Streamlit App
# st.set_page_config(layout="wide")

st.title('Delhi PM2.5 Prediction App 🌫️')

st.sidebar.header('Input Features')

def user_input_features():
    # Date and Time inputs
    d = st.sidebar.date_input("Date", datetime.date.today())
    t = st.sidebar.time_input("Time", datetime.time(12, 0))
    prediction_datetime = datetime.datetime.combine(d, t)

    # Pollutant inputs
    co = st.sidebar.slider('CO', float(df['co'].min()), float(df['co'].max()), float(df['co'].mean()))
    no = st.sidebar.slider('NO', float(df['no'].min()), float(df['no'].max()), float(df['no'].mean()))
    no2 = st.sidebar.slider('NO2', float(df['no2'].min()), float(df['no2'].max()), float(df['no2'].mean()))
    o3 = st.sidebar.slider('O3', float(df['o3'].min()), float(df['o3'].max()), float(df['o3'].mean()))
    so2 = st.sidebar.slider('SO2', float(df['so2'].min()), float(df['so2'].max()), float(df['so2'].mean()))
    pm10 = st.sidebar.slider('PM10', float(df['pm10'].min()), float(df['pm10'].max()), float(df['pm10'].mean()))
    nh3 = st.sidebar.slider('NH3', float(df['nh3'].min()), float(df['nh3'].max()), float(df['nh3'].mean()))
    
    # Get the latest row for lag and rolling mean features
    latest_row = df.iloc[-1]
    
    data = {
        'co': co,
        'no': no,
        'no2': no2,
        'o3': o3,
        'so2': so2,
        'pm10': pm10,
        'nh3': nh3,
        'hour': prediction_datetime.hour,
        'day_of_week': prediction_datetime.weekday(),
        'month': prediction_datetime.month,
        'day_of_year': prediction_datetime.timetuple().tm_yday,
        'pm2_5_lag_1': df['pm2_5'].iloc[-1],
        'pm2_5_lag_2': df['pm2_5'].iloc[-2],
        'pm2_5_lag_3': df['pm2_5'].iloc[-3],
        'pm2_5_lag_24': df['pm2_5'].iloc[-24],
        'pm2_5_rolling_mean_3': df['pm2_5'].iloc[-3:].mean(),
        'pm2_5_rolling_mean_24': df['pm2_5'].iloc[-24:].mean()
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)
st.info("Note: Lag and rolling mean features are based on the latest available historical data.")


prediction = reg.predict(input_df)

st.subheader('Prediction')
st.write(f'Predicted PM2.5: **{prediction[0]:.2f} µg/m³**')

# st.subheader('Historical Data Visualization')
# st.line_chart(df['pm2_5'])

st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'feature': reg.feature_names_in_,
    'importance': reg.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
st.pyplot(fig)
