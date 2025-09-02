import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Country Clustering", layout="wide")
st.title("🌍 Country Clustering Based on Development Indicators")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])
if not uploaded_file:
    st.stop()

# ------------------ LOAD AND CLEAN ------------------
df_raw = pd.read_excel(uploaded_file)
df = df_raw.copy()

# Identify country column
country_col = 'Country' if 'Country' in df.columns else df.columns[0]
country_names = df[country_col]
df.drop(columns=[country_col], inplace=True)

# Clean and convert columns
def clean_column(col):
    return pd.to_numeric(col.astype(str)
                         .str.replace('$','',regex=True)
                         .str.replace('%','',regex=True)
                         .str.replace(',','',regex=True),
                         errors='coerce')

df = df.apply(clean_column)

# Impute missing values
df.fillna(df.median(), inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Final cleanup before clustering
X_scaled = np.nan_to_num(X_scaled)

# ------------------ SIDEBAR SETTINGS ------------------
st.sidebar.header("🔧 Clustering Settings")
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

# ------------------ CLUSTERING ------------------
model = KMeans(n_clusters=n_clusters, random_state=42)
labels = model.fit_predict(X_scaled)

# ------------------ ASSIGN CLUSTERS ------------------
df['Country'] = country_names.reset_index(drop=True)
df['Cluster'] = labels

# ------------------ METRICS ------------------
if len(set(labels)) > 1:
    score = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{score:.3f}")
else:
    st.warning("Only one cluster detected. Try increasing variation or adjusting parameters.")

# ------------------ CLUSTER SUMMARY ------------------
st.subheader("📋 Cluster Summary")
numeric_cols = df.select_dtypes(include='number').columns.drop('Cluster')
summary = df.groupby('Cluster')[numeric_cols].mean().round(2)
st.dataframe(summary)

# ------------------ COUNTRIES BY CLUSTER ------------------
st.subheader("🌍 Countries by Cluster")
country_cluster_df = df[['Country', 'Cluster']].sort_values(by='Cluster')
st.dataframe(country_cluster_df)

# ------------------ OPTIONAL: CLUSTER DISTRIBUTION ------------------
st.subheader("📊 Cluster Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Cluster', data=df, palette='Set2', ax=ax)
ax.set_title("Number of Countries per Cluster")
st.pyplot(fig)
