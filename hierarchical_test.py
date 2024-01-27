# Import necessary libraries
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Read data from CSV file into a Pandas DataFrame
df = pd.read_csv('data_files/new_data.csv')

# Transpose the DataFrame to get features as rows
features = df.iloc[:, 1:].transpose().values.tolist()

# Extract the features as a list of tuples
data = list(zip(*features))

# Initialize variables to track the best clustering results
max_silhouette_avg = 0
best_n_clusters = 0
labels = []

# Iterate over a range of cluster numbers from 2 to 500
for i in range(2, 501):
    # Create an Agglomerative Clustering model with current cluster number
    hierarchical_cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')

    # Fit the model and predict cluster labels
    cluster_labels = hierarchical_cluster.fit_predict(data)

    # Calculate the silhouette score for the current clustering
    silhouette_avg = silhouette_score(data, cluster_labels)

    # Print silhouette score for current cluster number
    print("For no of clusters = ", i, " The average silhouette_score is: ", silhouette_avg)

    # Update best clustering if the current silhouette score is higher
    if silhouette_avg > max_silhouette_avg:
        max_silhouette_avg = silhouette_avg
        best_n_clusters = i
        labels = cluster_labels

# Print the final cluster labels and best clustering results
print(labels)
print("Clusters: ", best_n_clusters)
print("Silhouette score: ", max_silhouette_avg)

# Replace the column 'utility' of the original DataFrame with cluster labels
df['utility'] = labels

# Save the clustered data to a new CSV file
df.to_csv('data_files/clustered_data.csv', encoding='utf-8', index=False)
