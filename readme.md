# Rainfall Prediction Model

>While I started it as a data science/machine learning project ( *which I often find boring* ), it evolved into data engineering project ( *something I love to do!* ).


`rain-prediction.ipynb` is the kaggle notebook I downloaded after the data analysis part was done. Later I added more code in it.

I initially followed this [Geeks For Geeks Article](https://www.geeksforgeeks.org/rainfall-prediction-using-machine-learning-python/). While it was a helpful learning resource I followed, the dataset I used `weatherAUS.csv`  was different from the one in the article and required more data cleaning.

>As a result, I decided to take a DIY approach to data engineering for the project. I focused on cleaning and preparing the data to ensure it was ready for modeling.

## [app.py](/app.py)

>This file contains some of the data science code and all of the data engineeting work I did.

Here, I cleaned the data. Transformed it into the format I wished.

Since, this data contains daily weather information of entire Australia. I had to split it based on cities (Location column).

**You need not re-run this file.**

## [rain-prediction.ipynb](/rain-prediction.ipynb)

>This file has all the analysis and the model

>I've commented it well!

