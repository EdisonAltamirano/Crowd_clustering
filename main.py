
import pandas as pd
from sklearn.cluster import KMeans
import cv2
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

# from streamlit_gallery.utils.page import page_group

# import streamlit as st

def main():
    # page = page_group("p")
    page.item("Crowd Color Clustering", apps.gallery, default=True)
    # Load the image using OpenCV
    image_path = "bracelet_color.jpg"  # Replace with the path to your image
    #image_path = "test.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 1D array of pixels
    pixels = image.reshape(-1, 3)

    # Convert the pixel values to a DataFrame
    df = pd.DataFrame(pixels, columns=["Red", "Green", "Blue"])

    # Perform clustering on the pixel values
    num_clusters = 5  # Adjust the number of clusters as per your preference
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df)

    df['Cluster'] = kmeans.labels_
    # Count the number of pixels in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()
    # Compute the centroid colors of each cluster
    centroid_colors = kmeans.cluster_centers_.astype(int)
    cluster_colors = [tuple(color) for color in centroid_colors]

    # # Create a bar plot using Plotly
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig = go.Figure(data=[
        go.Bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            marker_color=[f"rgb{color}" for color in cluster_colors]
        )
    ])

    # Update the layout of the plot
    fig.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Count',
        title='Color Clusters'
    )

    # Show the plot
    fig.show()
    # page.show()
if __name__ == "__main__":
    st.set_page_config(page_title="Crowd clustering", page_icon="🤖", layout="wide")
    main()