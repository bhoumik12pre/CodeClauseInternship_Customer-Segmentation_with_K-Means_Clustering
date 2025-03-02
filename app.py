import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Streamlit UI
st.title("Customer Segmentation using K-Means")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.dataframe(df.head())
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['LIFESTAGE'] = label_encoder.fit_transform(df['LIFESTAGE'])
    df['PREMIUM_CUSTOMER'] = label_encoder.fit_transform(df['PREMIUM_CUSTOMER'])
    
    # Aggregate purchase data per customer
    df_grouped = df.groupby('LYLTY_CARD_NBR').agg({'LIFESTAGE': 'first', 'PREMIUM_CUSTOMER': 'first'}).reset_index()
    
    # Normalize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_grouped.drop(columns=['LYLTY_CARD_NBR']))
    
    # Determine optimal number of clusters using Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
    
    st.write("### Elbow Method for Optimal K")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method')
    st.pyplot(fig)
    
    # Select number of clusters
    k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_grouped['Cluster'] = kmeans.fit_predict(df_scaled)
    
    # Visualize Clusters using PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    df_grouped['PCA1'] = df_pca[:, 0]
    df_grouped['PCA2'] = df_pca[:, 1]
    
    st.write("### Customer Segments Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_grouped, palette='viridis', ax=ax)
    ax.set_title('Customer Segments')
    st.pyplot(fig)
    
    # Download clustered data
    st.write("### Download Segmented Data")
    csv = df_grouped.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "customer_segments.csv", "text/csv")
