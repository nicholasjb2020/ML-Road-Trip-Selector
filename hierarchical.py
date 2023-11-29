import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('CS4260/data_files/new_data.csv')
lol = []
lol.append(df['t0'].tolist())
lol.append(df['t1'].tolist())
lol.append(df['t2'].tolist())
lol.append(df['t3'].tolist())
lol.append(df['t4'].tolist())
lol.append(df['t5'].tolist())
lol.append(df['t6'].tolist())
lol.append(df['t7'].tolist())
lol.append(df['t8'].tolist())
lol.append(df['t9'].tolist())

data = list(zip(*lol))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

# plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
print(labels)