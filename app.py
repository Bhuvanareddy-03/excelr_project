import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
from io import BytesIO
from sklearn.impute import SimpleImputer

# Streamlit page setup
st.set_page_config(page_title="Country Clustering", layout="wide")
st.title("🌍 Country Clustering Based on Development Indicators")

# Upload Excel File
uploaded_file = st.file_uploader("📤 Upload Preprocessed Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Load Excel into DataFrame
        data = pd.read_excel(uploaded_file)

        st.subheader("📄 Preview of Uploaded Data")
        st.write(data.head())

        # Show missing data report
        st.subheader("🔍 Missing Value Report")
        st.write(data.isna().sum())

        # Select only numeric columns for clustering
        numeric_data = data.select_dtypes(include=[np.number])

        # Replace infinite values and drop NaNs
        numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optional: Impute missing values instead of dropping
        imputer = SimpleImputer(strategy="mean")
        numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data),
                                            columns=numeric_data.columns)

        # Sidebar clustering settings
        st.sidebar.header("⚙️ Clustering Settings")
        method = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "Hierarchical", "DBSCAN"])
        reduction = st.sidebar.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])

        if method in ["K-Means", "Hierarchical"]:
            n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        elif method == "DBSCAN":
            eps = st.sidebar.slider("DBSCAN: Epsilon", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            min_samples = st.sidebar.slider("DBSCAN: Min Samples", min_value=2, max_value=10, value=5)

        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data_imputed)

        # Dimensionality reduction
        st.subheader("📉 Dimensionality Reduction")
        if reduction == "PCA":
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(scaled_data)
            explained_var = pca.explained_variance_ratio_
            st.write(f"Explained Variance: PC1 = {explained_var[0]:.2f}, PC2 = {explained_var[1]:.2f}")
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            reduced = tsne.fit_transform(scaled_data)
            st.write("t-SNE applied for 2D projection.")

        # Clustering
        st.subheader("🧠 Clustering Results")
        if method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)

        labels = model.fit_predict(reduced)
        data['Cluster'] = labels  # Add cluster labels to original data

        # Silhouette Score
        if len(set(labels)) > 1 and -1 not in labels:
            score = silhouette_score(reduced, labels)
            st.success(f"Silhouette Score: {score:.3f}")
        elif len(set(labels)) > 1:
            filtered = labels != -1
            score = silhouette_score(reduced[filtered], np.array(labels)[filtered])
            st.warning(f"Silhouette Score (excluding noise): {score:.3f}")
        else:
            st.error("⚠️ Only one cluster found or all noise. Silhouette Score not applicable.")

        # Cluster visualization
        st.subheader("📊 Cluster Visualization")
        plot_df = pd.DataFrame(reduced, columns=["Component 1", "Component 2"])
        plot_df["Cluster"] = labels.astype(str)

        fig = px.scatter(
            plot_df,
            x="Component 1",
            y="Component 2",
            color="Cluster",
            title="Clusters in 2D Space",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show cluster sizes
        st.subheader("📌 Cluster Distribution")
        st.write(data['Cluster'].value_counts())

        # Preview data with cluster labels
        st.subheader("📋 Data with Cluster Labels")
        st.dataframe(data.head(20))

        # Prepare Excel download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='Clustered_Data')
        st.download_button(
            label="📥 Download Clustered Data (Excel)",
            data=output.getvalue(),
            file_name="clustered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
