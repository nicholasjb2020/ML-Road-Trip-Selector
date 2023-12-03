import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv('data_files/new_data.csv')
features = df.iloc[:, 1:].transpose().values.tolist()

data = list(zip(*features))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
silhouette_avg = silhouette_score(data, labels)
print(labels)
print("Silhouette score: ", silhouette_avg)
# interesting note: this silhouette score is much lower than that of our best result in kmeans
# however, if we ignore the dendrogram and instead use same number of clusters as we used for kmeans (338),
# then we get a significantly higher silhouette score with hierarchical
