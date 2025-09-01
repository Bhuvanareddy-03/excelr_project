# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

st.set_page_config(page_title="Country Clustering App", layout="wide")
st.title("🌍 Country Clustering Based on Development Indicators")

# Upload dataset
uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])
if uploaded_file:
    data_org = pd.read_excel(uploaded_file)
    data = data_org.copy()

    st.subheader("📊 Raw Data Preview")
    st.dataframe(data.head())

    # Encode country
    data['Country_encoded'] = LabelEncoder().fit_transform(data['Country'])
    country_names = data['Country']
    data.drop(['Country'], axis=1, inplace=True)

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

    # Impute missing values using mean or median based on skewness
    st.subheader("🧹 Handling Missing Values")
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['float64', 'int64']:
                skew = data[col].skew()
                if abs(skew) < 1:
                    data[col] = data[col].fillna(data[col].mean())
                    st.write(f"Filled missing values in '{col}' with mean.")
                else:
                    data[col] = data[col].fillna(data[col].median())
                    st.write(f"Filled missing values in '{col}' with median.")
            else:
                data[col] = data[col].fillna("Unknown")
                st.write(f"Filled missing values in '{col}' with placeholder.")

    # Final sweep to ensure no NaNs remain
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    st.write(f"✅ Remaining missing values: {data.isnull().sum().sum()}")

    # Outlier removal using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Re-impute zeros in key economic indicators
    st.subheader("🔄 Re-imputing zeros in key indicators")
    key_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
    for col in key_cols:
        if col in data_cleaned.columns:
            data_cleaned[col] = data_cleaned[col].replace(0, np.nan)
            non_zero = data_cleaned[col].dropna()
            if len(non_zero) > 0:
                if abs(non_zero.skew()) < 1:
                    data_cleaned[col] = data_cleaned[col].fillna(non_zero.mean())
                    st.write(f"Replaced zeros in '{col}' with mean.")
                else:
                    data_cleaned[col] = data_cleaned[col].fillna(non_zero.median())
                    st.write(f"Replaced zeros in '{col}' with median.")

    # Final cleanup before PCA
    st.subheader("🧼 Final Cleanup Before PCA")
    data_cleaned_numeric = data_cleaned.select_dtypes(include=[np.number])
    data_cleaned_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in data_cleaned_numeric.columns:
        if data_cleaned_numeric[col].isnull().sum() > 0:
            median_val = data_cleaned_numeric[col].median()
            data_cleaned_numeric[col] = data_cleaned_numeric[col].fillna(median_val)
            st.write(f"Filled NaNs in '{col}' with median: {median_val}")

    total_missing = data_cleaned_numeric.isnull().sum().sum()
    st.write("✅ Remaining missing values before PCA:", total_missing)

    if total_missing > 0:
        st.error("❌ PCA cannot proceed. Missing values still present.")
        st.stop()

    # Standardize and apply PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_cleaned_numeric)
    pca = PCA()
    data_pca = pca.fit_transform(scaled)
    data_pca_15 = data_pca[:, :15]

    # Sidebar options
    st.sidebar.header("🔧 Clustering Settings")
    cluster_method = st.sidebar.selectbox("Choose clustering method", ["K-Means", "Hierarchical", "DBSCAN"])
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 10, 5)

    # Apply clustering
    if cluster_method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data_pca_15)
    elif cluster_method == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = model.fit_predict(data_pca_15)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data_pca_15)

    # Add cluster labels
    data_cleaned['Cluster'] = labels
    data_cleaned['Country'] = country_names.loc[data_cleaned.index]

    # Silhouette Score
    if len(set(labels)) > 1:
        score = silhouette_score(data_pca_15, labels)
        st.metric("Silhouette Score", f"{score:.3f}")
    else:
        st.warning("DBSCAN detected only one cluster or noise. Silhouette Score not available.")

    # PCA Visualization
    st.subheader("📉 PCA Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data_pca_15[:, 0], y=data_pca_15[:, 1], hue=labels, palette='Set2', s=100, ax=ax)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Clusters in PCA Space")
    st.pyplot(fig)

    # t-SNE Visualization
    st.subheader("🌐 t-SNE Cluster Visualization")
    tsne = TSNE(random_state=42)
    tsne_input = data_cleaned_numeric.copy()
    tsne_input.drop(columns=['Cluster'], errors='ignore', inplace=True)
    data_tsne = tsne.fit_transform(tsne_input)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=labels, palette='Set1', s=100, ax=ax2)
    ax2.set_title("Clusters in t-SNE Space")
    st.pyplot(fig2)

    # Dendrogram (only for Hierarchical)
    if cluster_method == "Hierarchical":
        st.subheader("🌲 Dendrogram")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sch.dendrogram(sch.linkage(data_pca_15, method='ward'), ax=ax3)
        st.pyplot(fig3)

    # Cluster Summary
    st.subheader("📋 Cluster Summary")
    st.dataframe(data_cleaned.groupby('Cluster')[key_cols].mean())

    # Clustered Countries
    st.subheader("🌍 Countries by Cluster")
    st.dataframe(data_cleaned[['Country', 'Cluster']].sort_values(by='Cluster'))
