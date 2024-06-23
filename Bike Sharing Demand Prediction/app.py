import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
@st.cache
def load_data():
    return pd.read_csv('hour.csv')

# Main function
def main():
    st.title('Bike Sharing Dataset Analysis')

    # Load the data
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Loading data...done!')

    # Sidebar for selecting features
    st.sidebar.title('Select Features')
    features = st.sidebar.multiselect('Choose features', data.columns.tolist(), default=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'])

    # Remove the target column 'cnt' from the features list
    if 'cnt' in features:
        features.remove('cnt')

    # Feature selection
    X = data[features]
    y = data['cnt']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # User input for prediction
    st.sidebar.title('Predict Bike Count')
    input_data = {}
    for feature in features:
        input_data[feature] = st.sidebar.slider(f'{feature} ({data[feature].min()} - {data[feature].max()})', float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

    # Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_prediction = dt_model.predict(pd.DataFrame([input_data]))[0]
    dt_r2 = r2_score(y_test, dt_model.predict(X_test))  # Calculate R^2 score

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_prediction = rf_model.predict(pd.DataFrame([input_data]))[0]
    rf_r2 = r2_score(y_test, rf_model.predict(X_test))  # Calculate R^2 score

    # Display predictions and R^2 scores
    st.sidebar.subheader('Prediction Results')
    st.sidebar.write('Decision Tree Prediction:', dt_prediction)
    st.sidebar.write('Decision Tree R^2 Score:', dt_r2)
    st.sidebar.write('Random Forest Prediction:', rf_prediction)
    st.sidebar.write('Random Forest R^2 Score:', rf_r2)

    # Display bike counts
    st.write('### Bike Counts')
    st.write(data['cnt'])

    # Visualization - Histogram
    st.write('### Histogram of Bike Counts')
    plt.figure(figsize=(12, 6))
    sns.histplot(data['cnt'], bins=30, kde=True)
    st.pyplot()

    # Visualization - Scatter Plot
    st.write('### Scatter Plot of Temperature vs. Bike Counts')
    st.scatter_chart(data=data, x='temp', y='cnt')

    # Visualization - Line Plot
    st.write('### Line Plot of Bike Counts over Time')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='dteday', y='cnt')
    plt.xticks(rotation=45)
    st.pyplot()

   

if __name__ == '__main__':
    main()
