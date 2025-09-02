import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🌍 Country Clustering Based on Development Indicators")

# File uploader (now accepts Excel)
uploaded_file = st.file_uploader("Upload preprocessed Excel file", type=["xlsx"])

if uploaded_file:
    # Load Excel
    data = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.write(data.head())

    # Sidebar parameters
    st.sidebar.header("Clustering Settings")
    method = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "Hierarchical", "DBSCAN"])
    reduction = st.sidebar.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])
    
    if method in ["K-Means", "Hierarchical"]:
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    elif method == "DBSCAN":
        eps = st.sidebar.slider("DBSCAN: Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("DBSCAN: Min Samples", 2, 10, 5)

    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)

    # Dimensionality reduction
    if reduction == "PCA":
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaled)
        st.subheader("📉 PCA Explained Variance")
        st.write(f"Component 1: {pca.explained_variance_ratio_[0]:.2f}, Component 2: {pca.explained_variance_ratio_[1]:.2f}")
    else:
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(scaled)

    # Clustering
    if method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(reduced)
    elif method == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(reduced)
    elif method == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(reduced)

    # Silhouette score
    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(reduced, labels)
        st.success(f"Silhouette Score: {score:.3f}")
    elif len(set(labels)) > 1:
        filtered = labels != -1
        score = silhouette_score(reduced[filtered], np.array(labels)[filtered])
        st.warning(f"Silhouette Score (excluding noise): {score:.3f}")
    else:
        score = None
        st.error("Clustering resulted in only one cluster or all noise.")

    # Add cluster labels
    data['Cluster'] = labels
    reduced_df = pd.DataFrame(reduced, columns=["Component 1", "Component 2"])
    reduced_df["Cluster"] = labels.astype(str)

    # Visualization
    fig = px.scatter(
        reduced_df,
        x="Component 1", y="Component 2",
        color="Cluster",
        title="Cluster Visualization",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster distribution
    st.subheader("📊 Cluster Distribution")
    st.write(data['Cluster'].value_counts())

    # Data preview
    st.subheader("🧾 Data with Cluster Labels")
    st.dataframe(data.head(20))

    # Download as Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Clustered Data')
    st.download_button(
        label="📥 Download Clustered Data (Excel)",
        data=output.getvalue(),
        file_name="clustered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
