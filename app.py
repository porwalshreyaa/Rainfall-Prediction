# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import RandomOverSampler

import os
import warnings 
warnings.filterwarnings('ignore')

# plt.ion()
df = pd.read_csv('data/weatherAUS.csv')

# Raw Data

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe().T)


# Data Cleaning

df = df.drop_duplicates()


for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'float64':
            val = df[col].mean()
            df[col] = df[col].fillna(val)
        else:
            df = df.dropna(subset=df.select_dtypes(include=['object']).columns)


# Wind Directions can be converted into float

wind_dir = { 'N': 0,
    'NNE': 22.5,
    'NE': 45,
    'ENE': 67.5,
    'E': 90,
    'ESE': 112.5,
    'SE': 135,
    'SSE': 157.5,
    'S': 180,
    'SSW': 202.5,
    'SW': 225,
    'WSW': 247.5,
    'W': 270,
    'WNW': 292.5,
    'NW': 315,
    'NNW': 337.5}

df = df.drop(columns=['Date', 'Rainfall', 'RainTomorrow'])

# Avg Wind Speed
df['WindSpeed'] = df[['WindSpeed3pm', 'WindSpeed9am']].mean(axis=1)
df = df.drop(columns = ['WindSpeed3pm', 'WindSpeed9am'])

df['WindGustDir'] = df['WindGustDir'].map(wind_dir)


df['WindDir9am'] = df['WindDir9am'].map(wind_dir)
df['WindDir3pm'] = df['WindDir3pm'].map(wind_dir)
# Avg Wind Direction
df['WindDir'] = df[['WindDir9am', 'WindDir3pm']].mean(axis=1)
df = df.drop(columns = ['WindDir9am', 'WindDir3pm'])

# Avg Humidity
df['Humidity'] = df[['Humidity9am', 'Humidity3pm']].mean(axis=1)
df = df.drop(columns = ['Humidity9am', 'Humidity3pm'])

# Avg Cloud
df['Cloud'] = df[['Cloud9am', 'Cloud3pm']].mean(axis=1)
df = df.drop(columns = ['Cloud9am', 'Cloud3pm'])

# Avg Temp
df['Temp'] = df[['Temp9am', 'Temp3pm']].mean(axis=1)
df = df.drop(columns = ['Temp9am', 'Temp3pm'])

# Avg Pressure
df['Pressure'] = df[['Pressure9am', 'Pressure3pm']].mean(axis=1)
df = df.drop(columns = ['Pressure9am', 'Pressure3pm'])



# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe().T)

def circular_diff(deg1, deg2):
    diff = (deg1 - deg2 + 180) % 360 - 180
    return diff

# Sudden Change in Wind direction during Gust
df['AirDirChange'] = df.apply(lambda row: circular_diff(row['WindGustDir'], row['WindDir']), axis=1)
df = df.drop(columns = ['WindGustDir'])


# Sudden Change in Wind speed during Gust
df['AirSpeedChange'] = df.apply(lambda row: circular_diff(row['WindGustSpeed'], row['WindSpeed']), axis=1)
df = df.drop(columns = ['WindGustSpeed'])

# print(df[['AirSpeedChange', 'WindSpeed', 'WindDir', 'AirDirChange']].head())

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe().T)


# Model trained on data of one city might not give accurate results for other cities
# because of geographical factors that we did not take into consideration
# Hence, We'll split this based on locations and save into folder called cities


# Splitting the Dataset

grouped_data = df.groupby('Location')
# print(grouped_data.count())
# Yeppie! The data has no anomoly
dire = 'data_cities/' 


# create csv files and save datesets into cities
# for location, group in grouped_data:

#     filename = os.path.join(dire, f'{location}.csv')
#     group.to_csv(filename, index=False)

#     print(f'Saved data for location {location} to file {filename}')


print('Your Datasets are ready, garnish them with corriander leaves... serve them fresh and hot!')