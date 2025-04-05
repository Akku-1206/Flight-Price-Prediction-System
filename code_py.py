# -*- coding: utf-8 -*-
Importing the libraries


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
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

"""# Loading the data"""

df = pd.read_csv("/content/goibibo_flights_data.csv")
df

"""Check for missing value"""

df.isnull().sum()

"""Drop column Unnamed: 11 and  Unnamed: 12"""

df = df.drop(['Unnamed: 11', 'Unnamed: 12'], axis=1)
df

df=df.dropna(axis=1)
df

df.dtypes

"""Extract Month from Date

Extract Hours from Departure Time

Convert duration to Minutes

Clean "Stops" Column
"""

# convert_to_minutes( x ) : This function will convert duration column to minutes
# clean_stops( x ) : This function will return Number of Stops of the flight

def convert_to_minutes(x):
    hours, minutes = x.split(" ")
    hours = hours.replace("h","")
    minutes = minutes.replace("m", "")
    if len(hours)==0:
        hours = 0
    elif len(minutes)==0:
        minutes =0
    hours = float(hours)
    minutes = float(minutes)
    return hours*60+minutes
def clean_stops(x):
    if 'non-stop' in x:
        return 0
    elif '1-' in x:
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

"""Apply Transformations in the columns"""

df['flight date'] = pd.to_datetime(df['flight date'])
df['duration'] = df['duration'].apply(lambda s:convert_to_minutes(s))

df

"""clean stop columns"""

df['stops']=df['stops'].apply(lambda s:clean_stops(s))

df

"""Extraction of Month from Flight Date


extracting Month from Flight Date as this will help me understand in which month prices spike
Price column as a comma, we will  remove it and convert it into Float
"""

# Extraction of Month from Flight Date

df['flight date']=pd.to_datetime(df['flight date'])
df['Month']=df['flight date'].dt.month
df['price']=df['price'].apply(lambda s:float(s.replace(',','')))
df

"""Drop Unnecessary Columns


Drop "From" and "To" Column as Price does not depend on cities

Drop Departure Time and Arrival Time because we have the flight duration column

Remove Flight Number Column because we have airline column
"""

# drop unnecessay columns
df.drop(columns=['from','to','dep_time','arr_time','flight date','flight_num'],inplace=True)

df

df2=df.groupby('airline').agg({"price":"mean"}).reset_index().sort_values(by='price')


sns.barplot(x='airline',y='price',data=df2)

# price trends over months

df2=df.groupby(["Month"]).agg({"price":"mean"}).reset_index().sort_values(by='price')
sns.lineplot(x='Month',y='price',data=df2)

sns.pairplot(df2[['price',"Month"]])

"""Direct Flights are cheapest may be due to low distance

Flights with 1 stop are expensive
"""

# Impact of class on Price

df2=df.groupby(["airline","class"]).agg({"price":"mean"}).reset_index().sort_values(by='price')
sns.barplot(x='class',y='price',data=df2,hue='airline')

# Checkig correaltion values
sns.heatmap(df[['duration', 'price', 'stops', 'Month']].corr().abs(),annot=True)

# Data Standardization and Encoding

#I will encode Airline and Class Columns using OneHotEncoding
#I will standardize numerical columns -'Month','stops','duration'

X=df.drop(columns='price')
Y=df['price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

transformer = ColumnTransformer(
[
    ("encode",OneHotEncoder(),['airline','class']),
    ("standardise",StandardScaler(),['Month','stops','duration'])

],remainder='passthrough')

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Model Training and Evaluation

models = {"Random Forest Regressor":RandomForestRegressor(),
         "Decision Tree Regressor":DecisionTreeRegressor(),
         "Linear Regressor":LinearRegression(),
         "XGBoost Regression":XGBRegressor()}

r2_scores=[]
mean_squared = []
mean_absolute = []
model_list =[]
for model_name,model in models.items():
    model.fit(X_train_transformed,Y_train)
    ypred = model.predict(X_test_transformed)
    r2_scores.append(r2_score(Y_test,ypred))
    mean_squared.append(mean_squared_error(Y_test,ypred))
    mean_absolute.append(mean_absolute_error(Y_test,ypred))
    model_list.append(model_name)

accuracy_data = pd.DataFrame()
accuracy_data['Model ']=model_list
accuracy_data['R2 Score']=r2_scores

accuracy_data

