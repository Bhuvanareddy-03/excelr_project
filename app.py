import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

st.set_page_config(page_title="Country Clustering Diagnostics", layout="wide")
st.title("🔍 Country Clustering Diagnostics")

uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])
if not uploaded_file:
    st.stop()

df_raw = pd.read_excel(uploaded_file)
df = df_raw.copy()

# Identify country column
country_col = 'Country' if 'Country' in df.columns else df.columns[0]
country_names = df[country_col]
df.drop(columns=[country_col], inplace=True)

# Clean and convert
def clean_column(col):
    return pd.to_numeric(col.astype(str)
                         .str.replace('$','',regex=True)
                         .str.replace('%','',regex=True)
                         .str.replace(',','',regex=True),
                         errors='coerce')

df = df.apply(clean_column)
df.fillna(df.median(), inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# Show feature variance
st.subheader("📊 Feature Variance")
variance = df.var().round(4)
st.dataframe(variance)

# Show distributions for key indicators
key_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
st.subheader("📈 Key Indicator Distributions")
for col in key_cols:
    if col in df.columns:
        st.write(f"{col} Summary:")
        st.dataframe(df[col].describe().round(2))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled = np.nan_to_num(X_scaled)

# Clustering
model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X_scaled)

# Show cluster label counts
st.subheader("🔢 Cluster Label Counts")
label_counts = pd.Series(labels).value_counts().sort_index()
st.dataframe(label_counts)

# Assign clusters
df['Cluster'] = labels
df['Country'] = country_names.reset_index(drop=True)

# Silhouette Score
if len(set(labels)) > 1:
    score = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{score:.3f}")
else:
    st.warning("Only one cluster detected. Try increasing variation or adjusting parameters.")

# t-SNE Visualization
st.subheader("📉 t-SNE Visualization")
tsne = TSNE(random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
fig, ax = plt.subplots()
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='Set2', s=100, ax=ax)
ax.set_title("Clusters in t-SNE Space")
st.pyplot(fig)

# Cluster Summary
st.subheader("📋 Cluster Summary")
numeric_cols = df.select_dtypes(include='number').columns.drop('Cluster')
summary = df.groupby('Cluster')[numeric_cols].mean().round(2)
st.dataframe(summary)

# Countries by Cluster
st.subheader("🌍 Countries by Cluster")
country_cluster_df = df[['Country', 'Cluster']].sort_values(by='Cluster')
st.dataframe(country_cluster_df)
