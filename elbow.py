import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# Load the IRIS dataset
data = pd.read_csv('data_files/new_data.csv')

X = data.iloc[:, 1:]

fig, ax = plt.subplots(5, 2, figsize=(15, 8))
for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)

    # Fit KMeans
    km.fit(X)

    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])

    # Fit the visualizer
    visualizer.fit(X)

    # Get and print the silhouette score
    silhouette_avg = silhouette_score(X, km.labels_)
    print(f"For n_clusters = {i}, the average silhouette_score is : {silhouette_avg}")

plt.show()
