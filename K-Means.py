import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
rssi_data = pd.read_csv("rssi_data4.csv")

# Extract the rssi values
rssi_values = rssi_data[['rssi']]

# Apply K-means clustering with K=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(rssi_values)

# Extract the centroids and sort them in descending order
sorted_centroids = sorted(kmeans.cluster_centers_.flatten(), reverse=True)

# Calculate the RSSI thresholds
t_values = [(sorted_centroids[i-1] + sorted_centroids[i]) / 2 for i in range(1, 3)]

print("Centroids:", sorted_centroids)
print("RSSI Thresholds:", t_values)
