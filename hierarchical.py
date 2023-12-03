import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv('data_files/new_data.csv')
features = df.iloc[:, 1:].transpose().values.tolist()

data = list(zip(*features))

max_silhouette_avg = 0
best_n_clusters = 0
labels = []
for i in range(2, 900):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
    cluster_labels = hierarchical_cluster.fit_predict(data)
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

df['utility'] = labels
df.to_csv('data_files/clustered_data.csv', encoding='utf-8', index=False)
