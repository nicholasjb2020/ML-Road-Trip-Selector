# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from yellowbrick.cluster import SilhouetteVisualizer
#
# # Load the IRIS dataset
# data = pd.read_csv('data_files/new_data.csv')
#
# X = data.iloc[:, 1:]
#
# max = 0
# num = 2
#
# fig, ax = plt.subplots(6, 2, figsize=(15, 8))
# for i in range(2, 1000):
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
#
#     # Fit KMeans
#     km.fit(X)
#
#     q, mod = divmod(i, 2)
#    # visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
#
#     # Fit the visualizer
#    # visualizer.fit(X)
#
#     # Get and print the silhouette score
#     silhouette_avg = silhouette_score(X, km.labels_)
#
#     if silhouette_avg > max:
#         max = silhouette_avg
#         num = i
#
# print(f"The optimal number is {num} with score {max}"