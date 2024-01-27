# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Read data from CSV file into a Pandas DataFrame
df = pd.read_csv('data_files/new_data.csv')

# Extract numerical data from DataFrame and convert it to a list
data = df.iloc[:, 1:].values.tolist()

# Initialize variables to track the best clustering parameters and results
max_silhouette_avg = 0
best_n_clusters = 0
labels = []

# Iterate through a range of cluster numbers to find the optimal number of clusters
for i in range(2, 501):
    # Create a KMeans clustering model with the current number of clusters
    kmeans = KMeans(n_clusters=i, random_state=2)

    # Fit the model to the data and predict cluster labels
    cluster_labels = kmeans.fit_predict(data)

    # Calculate the silhouette score for the current clustering
    silhouette_avg = silhouette_score(data, cluster_labels)

    # Print the silhouette score for the current iteration
    print("For no of clusters =", i,
          "The average silhouette_score is:", silhouette_avg)

    # Update the best clustering parameters if the current result is better
    if silhouette_avg > max_silhouette_avg:
        max_silhouette_avg = silhouette_avg
        best_n_clusters = i
        labels = cluster_labels

# Print the final cluster labels and the best clustering parameters
print(labels)
print("Clusters:", best_n_clusters)
print("Silhouette score:", max_silhouette_avg)
