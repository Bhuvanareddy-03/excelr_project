import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Country Clustering", layout="wide")
st.title("🌍 Country Clustering Based on Development Indicators")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader('World_development_mesurement.xlsx', type=["xlsx"])
if not uploaded_file:
    st.stop()

df_raw = pd.read_excel(uploaded_file)
df = df_raw.copy()
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head())

# ------------------ PREPROCESSING ------------------
def clean_and_impute(df):
    df = df.copy()
    if 'Country' in df.columns:
        countries = df['Country']
        df.drop(columns=['Country'], inplace=True)
    else:
        countries = pd.Series([f"Country_{i}" for i in range(len(df))])

    # Clean currency and percentage columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('$','',regex=True).str.replace('%','',regex=True).str.replace(',','',regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Remove outliers (IQR)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Final cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    return df, countries.loc[df.index].reset_index(drop=True)

df_clean, country_names = clean_and_impute(df)

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# ------------------ SIDEBAR SETTINGS ------------------
st.sidebar.header("🔧 Clustering Settings")
method = st.sidebar.selectbox("Method", ["K-Means", "Hierarchical", "DBSCAN"])
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5)
min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 10, 5)

# ------------------ CLUSTERING ------------------
if method == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
elif method == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
else:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

df_clean['Cluster'] = labels
df_clean['Country'] = country_names

# ------------------ METRICS ------------------
valid_clusters = [label for label in set(labels) if label != -1]
if len(valid_clusters) > 1:
    score = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{score:.3f}")

# ------------------ VISUALIZATION ------------------
st.subheader("📉 t-SNE Cluster Visualization")
tsne = TSNE(random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
fig, ax = plt.subplots()
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='Set2', s=100, ax=ax)
ax.set_title("Clusters in t-SNE Space")
st.pyplot(fig)

if method == "Hierarchical":
    st.subheader("🌲 Dendrogram")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sch.dendrogram(sch.linkage(X_scaled, method='ward'), ax=ax2)
    st.pyplot(fig2)

# ------------------ OUTPUT ------------------
st.subheader("📋 Cluster Summary")
summary = df_clean.groupby('Cluster').mean().round(2)
st.dataframe(summary)

st.subheader("🌍 Countries by Cluster")
country_cluster = df_clean[['Country', 'Cluster']].sort_values(by='Cluster')
country_cluster = country_cluster[country_cluster['Cluster'] >= 0]
st.dataframe(country_cluster)
