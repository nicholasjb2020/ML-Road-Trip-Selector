import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('data_files/new_data.csv')

data = df.iloc[:, 1:].values.tolist()

max_silhouette_avg = 0
best_n_clusters = 0
labels = []
for i in range(2, 900):
    kmeans = KMeans(n_clusters=i, random_state=2)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    print("For no of clusters = ", i,
          " The average silhouette_score is: ", silhouette_avg)
    if silhouette_avg > max_silhouette_avg:
        max_silhouette_avg = silhouette_avg
        best_n_clusters = i
        labels = cluster_labels
print(labels)
print("Clusters: ", best_n_clusters)
print("Silhouette score: ", max_silhouette_avg)
