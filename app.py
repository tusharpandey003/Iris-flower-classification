import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris FLower classification")
st.write("""
# IRIS FLOWER PREDICTION APP
         
This app predicts the **iris flower** type!
""")
st.sidebar.header('user input Parameters')


def user_input():
    sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input()

st.subheader('user_input_parameters')
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(x, y)

pred = clf.predict(df)
predprob = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[pred])

st.subheader('Prediction Probablity')
st.write(predprob)
