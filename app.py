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
uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])
if not uploaded_file:
    st.stop()

df_raw = pd.read_excel(uploaded_file)
df = df_raw.copy()

# ------------------ COUNTRY COLUMN ------------------
country_col = 'Country' if 'Country' in df.columns else df.columns[0]
country_names = df[country_col]
df.drop(columns=[country_col], inplace=True)

# ------------------ CLEANING ------------------
def clean_column(col):
    return pd.to_numeric(col.astype(str)
                         .str.replace('$','',regex=True)
                         .str.replace('%','',regex=True)
                         .str.replace(',','',regex=True),
                         errors='coerce')

df = df.apply(clean_column)
df.fillna(df.median(), inplace=True)

# ------------------ OUTLIER REMOVAL ------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[mask]
country_names = country_names.loc[df.index].reset_index(drop=True)
df.reset_index(drop=True, inplace=True)

# ------------------ FINAL CLEANUP ------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled = np.nan_to_num(X_scaled)

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

df['Cluster'] = labels
df['Country'] = country_names

# ------------------ METRICS ------------------
valid_labels = sorted(label for label in set(labels) if label >= 0)
if len(valid_labels) > 1:
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
numeric_cols = df.select_dtypes(include='number').columns.drop('Cluster')
summary = df[df['Cluster'] >= 0].groupby('Cluster')[numeric_cols].mean().round(2)
st.dataframe(summary)

st.subheader("🌍 Countries by Cluster")
country_cluster_df = df[df['Cluster'] >= 0][['Country', 'Cluster']].sort_values(by='Cluster')
st.dataframe(country_cluster_df)
