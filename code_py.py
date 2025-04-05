import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide")
st.title("Goibibo Flight Price Prediction App ✈️")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Data Cleaning
    df = df.drop(['Unnamed: 11', 'Unnamed: 12'], axis=1, errors='ignore')
    df = df.dropna(axis=1)

    # Convert Duration to Minutes
    def convert_to_minutes(x):
        try:
            hours, minutes = x.split(" ")
            hours = hours.replace("h", "")
            minutes = minutes.replace("m", "")
        except:
            hours = x.replace("h", "")
            minutes = "0"
        hours = float(hours) if hours else 0
        minutes = float(minutes) if minutes else 0
        return hours * 60 + minutes

    def clean_stops(x):
        if 'non-stop' in x:
            return 0
        elif '1' in x:
            return 1
        elif '2' in x:
            return 2
        elif '3' in x:
            return 3
        elif '4' in x:
            return 4
        elif '5' in x:
            return 5
        elif '6' in x:
            return 6
        else:
            return np.nan

    df['flight date'] = pd.to_datetime(df['flight date'])
    df['duration'] = df['duration'].apply(convert_to_minutes)
    df['stops'] = df['stops'].apply(clean_stops)
    df['Month'] = df['flight date'].dt.month
    df['price'] = df['price'].apply(lambda s: float(str(s).replace(',', '')))

    df.drop(columns=['from', 'to', 'dep_time', 'arr_time', 'flight date', 'flight_num'], inplace=True, errors='ignore')

    st.subheader("Cleaned Data Sample")
    st.dataframe(df.head())

    # EDA section
    st.subheader("📊 Exploratory Data Analysis")
    eda_option = st.selectbox("Choose EDA Plot", ["Airline vs Price", "Price over Months", "Class Impact", "Correlation Heatmap"])

    if eda_option == "Airline vs Price":
        df2 = df.groupby('airline').agg({"price": "mean"}).reset_index().sort_values(by='price')
        fig = plt.figure(figsize=(10,5))
        sns.barplot(x='airline', y='price', data=df2)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif eda_option == "Price over Months":
        df2 = df.groupby(["Month"]).agg({"price": "mean"}).reset_index()
        fig = plt.figure()
        sns.lineplot(x='Month', y='price', data=df2, marker="o")
        st.pyplot(fig)

    elif eda_option == "Class Impact":
        df2 = df.groupby(["airline", "class"]).agg({"price": "mean"}).reset_index()
        fig = plt.figure(figsize=(10,5))
        sns.barplot(x='class', y='price', hue='airline', data=df2)
        st.pyplot(fig)

    elif eda_option == "Correlation Heatmap":
        fig = plt.figure()
        sns.heatmap(df[['duration', 'price', 'stops', 'Month']].corr().abs(), annot=True)
        st.pyplot(fig)

    # Modeling
    st.subheader("🤖 Model Training & Evaluation")

    if st.button("Train Models"):
        X = df.drop(columns='price')
        Y = df['price']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        transformer = ColumnTransformer([
            ("encode", OneHotEncoder(), ['airline', 'class']),
            ("standardise", StandardScaler(), ['Month', 'stops', 'duration'])
        ], remainder='passthrough')

        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        models = {
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Linear Regressor": LinearRegression(),
            "XGBoost Regression": XGBRegressor()
        }

        results = []

        for model_name, model in models.items():
            model.fit(X_train_transformed, Y_train)
            y_pred = model.predict(X_test_transformed)

            results.append({
                "Model": model_name,
                "R2 Score": round(r2_score(Y_test, y_pred), 4),
                "MSE": round(mean_squared_error(Y_test, y_pred), 2),
                "MAE": round(mean_absolute_error(Y_test, y_pred), 2)
            })

        st.subheader("📈 Model Performance")
        st.dataframe(pd.DataFrame(results).set_index("Model"))
