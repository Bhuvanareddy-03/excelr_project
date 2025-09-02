import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

st.set_page_config(page_title="Country Clustering App", layout="wide")
st.title("🌍 Country Clustering Based on Development Indicators")

uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])
if uploaded_file:
    # Load and preserve original country names
    data_org = pd.read_excel(uploaded_file)
    data = data_org.copy()
    country_names = data_org['Country']

    st.subheader("📊 Raw Data Preview")
    st.dataframe(data.head())

    # Clean currency and percentage columns
    def clean_currency(col):
        return pd.to_numeric(col.astype(str).str.replace('$','',regex=True).str.replace(',',''), errors='coerce')

    currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
    for col in currency_cols:
        if col in data.columns:
            data[col] = clean_currency(data[col])

    if 'Business Tax Rate' in data.columns:
        data['Business Tax Rate'] = pd.to_numeric(data['Business Tax Rate'].astype(str).str.replace('%','',regex=True), errors='coerce')

    if 'Number of Records' in data.columns:
        data.drop(['Number of Records'], axis=1, inplace=True)

    # Impute missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            skew = data[col].skew()
            if abs(skew) < 1:
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].median())

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Preserve key indicators during outlier removal
    key_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
    data_numeric = data.select_dtypes(include=[np.number])
    non_key = data_numeric.drop(columns=[col for col in key_cols if col in data_numeric.columns])

    Q1 = non_key.quantile(0.25)
    Q3 = non_key.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((non_key < (Q1 - 1.5 * IQR)) | (non_key > (Q3 + 1.5 * IQR))).any(axis=1)

    data_cleaned = data.loc[mask].copy()
    country_cleaned = country_names.loc[mask].reset_index(drop=True)
    data_cleaned.reset_index(drop=True, inplace=True)
    data_cleaned['Country'] = country_cleaned

    # Re-impute zeros in key indicators with fallback only if column is empty
    fallbacks = {'GDP': 1000, 'Health Exp/Capita': 50, 'Tourism Inbound': 100, 'Tourism Outbound': 100}
    for col in key_cols:
        if col in data_cleaned.columns:
            data_cleaned[col] = data_cleaned[col].replace(0, np.nan)
            if data_cleaned[col].isnull().all():
                data_cleaned[col] = data_cleaned[col].fillna(fallbacks[col])
            else:
                data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].median())

    # Final cleanup before clustering
    numeric_data = data_cleaned.select_dtypes(include=[np.number])
    numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in numeric_data.columns:
        if numeric_data[col].isnull().sum() > 0:
            median_val = numeric_data[col].median()
            numeric_data[col] = numeric_data[col].fillna(median_val)

    # Standardize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)

    # Final check before clustering
    scaled_df = pd.DataFrame(scaled)
    scaled_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    scaled_df.fillna(0, inplace=True)
    scaled = scaled_df.values

    # Sidebar clustering settings
    st.sidebar.header("🔧 Clustering Settings")
    cluster_method = st.sidebar.selectbox("Choose clustering method", ["K-Means", "Hierarchical", "DBSCAN"])
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 10, 5)

    # Apply clustering
    if cluster_method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(scaled)
    elif cluster_method == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = model.fit_predict(scaled)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(scaled)

    data_cleaned['Cluster'] = labels

    # Silhouette Score
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(scaled, labels)
        st.metric("Silhouette Score", f"{score:.3f}")

    # t-SNE Visualization
    st.subheader("🌐 t-SNE Cluster Visualization")
    tsne = TSNE(random_state=42)
    data_tsne = tsne.fit_transform(scaled)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=labels, palette='Set1', s=100, ax=ax2)
    ax2.set_title("Clusters in t-SNE Space")
    st.pyplot(fig2)

    # Dendrogram (only for Hierarchical)
    if cluster_method == "Hierarchical":
        st.subheader("🌲 Dendrogram")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sch.dendrogram(sch.linkage(scaled, method='ward'), ax=ax3)
        st.pyplot(fig3)

    # Cluster Summary
    st.subheader("📋 Cluster Summary")
    summary = data_cleaned.groupby('Cluster')[key_cols].mean().round(2)
    st.dataframe(summary)

    # Countries by Cluster
    st.subheader("🌍 Countries by Cluster")
    country_cluster_df = data_cleaned[['Country', 'Cluster']].sort_values(by='Cluster')
    country_cluster_df = country_cluster_df[country_cluster_df['Cluster'] >= 0]
    st.dataframe(country_cluster_df)
