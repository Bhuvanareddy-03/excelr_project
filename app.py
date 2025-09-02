import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Simple Country Clustering", layout="wide")
st.title("🌍 Simple Country Clustering")

uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])
if not uploaded_file:
    st.stop()

# Load and prepare data
df_raw = pd.read_excel(uploaded_file)
df = df_raw.copy()

# Identify country column
country_col = 'Country' if 'Country' in df.columns else df.columns[0]
country_names = df[country_col]
df.drop(columns=[country_col], inplace=True)

# Clean numeric columns
df = df.apply(lambda col: pd.to_numeric(col.astype(str)
                                        .str.replace('$','',regex=True)
                                        .str.replace('%','',regex=True)
                                        .str.replace(',','',regex=True),
                                        errors='coerce'))

df.fillna(df.median(), inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Clustering
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
model = KMeans(n_clusters=n_clusters, random_state=42)
labels = model.fit_predict(X_scaled)

# Assign clusters
df['Country'] = country_names.reset_index(drop=True)
df['Cluster'] = labels

# Silhouette Score
if len(set(labels)) > 1:
    score = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{score:.3f}")
else:
    st.warning("Only one cluster detected. Try increasing variation or adjusting parameters.")

# Show results
st.subheader("📋 Cluster Summary")
summary = df.groupby('Cluster').mean().round(2)
st.dataframe(summary)

st.subheader("🌍 Countries by Cluster")
country_cluster_df = df[['Country', 'Cluster']].sort_values(by='Cluster')
st.dataframe(country_cluster_df)
