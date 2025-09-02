#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
import pickle
import pandas as pd


# Load Train Kmeans Model
kmeans = pickle.load(open("kmeans.pkl",'rb'))
df = pd.read_excel("World_development_mesurement.xlsx")

# Simple clustering function
def clustering(GDP, emission, iu, eu):
    new_value = np.array([[GDP, emission, iu, eu]])
    predicted_cluster = kmeans.predict(new_value)

    if predicted_cluster[0] == 0:
        return "Developing Nations 🚧"
    elif predicted_cluster[0] == 1:
        return "Developed Nations  🌟"
    else:
        return "Emerging Economies or Developed Nations with Advanced Infrastructure 🌍"


# Streamlit app here==========================================
st.title("WORLD DEVELOPMENT MEASUREMENT PREDICTION 🌍")
st.write("Enter the details:")

# User input (side by side inputs

# row 1 with column 2
col1, col2 = st.columns(2)
with col1:
    st.subheader("GDP")
    GDP = st.number_input("GDP", min_value=18, max_value=100, value=40)

with col2:
    st.subheader("C02 Emissions")
    emission = st.number_input("C02 Emissions", min_value=0.0, max_value=1000.0, value=30.0)

# row 2 with column 2
col1, col2 = st.columns(2)
with col1:
    st.subheader("Internet Usage")
    iu = st.number_input("Internet Usage", min_value=0.0, max_value=1.0, value=0.5)

with col2:
    st.subheader("Energy Usage")
    eu = st.number_input("Energy Usage", min_value=0, max_value=10, value=7)
    
st.subheader("Country") 
country = st.selectbox("Select Country", df['Country'].unique())


# Predict button
if st.button("Predict Cluster"):
    cluster_label = clustering(GDP, emission, iu, eu)
 
    st.success(f'The selected country, {country}, belongs to the "{cluster_label}" cluster.')



# In[ ]:
