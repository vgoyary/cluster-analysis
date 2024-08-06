import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cluster data
cluster_0 = pd.read_csv('cluster_0.csv')
cluster_1 = pd.read_csv('cluster_1.csv')
cluster_2 = pd.read_csv('cluster_2.csv')

# Combine the data into one DataFrame for analysis
clusters = pd.concat([cluster_0, cluster_1, cluster_2])
clusters['Cluster'] = clusters['Cluster'].astype(int)

# Display descriptive statistics for each cluster
cluster_descriptions = clusters.groupby('Cluster').describe()
print(cluster_descriptions)

# Visualize the distribution of each feature in each cluster
features = clusters.columns[:-1]

for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=feature, data=clusters)
    plt.title(f'Boxplot of {feature} by Cluster')
    plt.show()

# Correlation matrix
corr = clusters.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()


# Calculate mean values of each feature for each cluster
cluster_means = clusters.groupby('Cluster').mean()
print(cluster_means)

# Plot the mean values
cluster_means.plot(kind='bar', figsize=(14, 8))
plt.title('Mean Values of Features by Cluster')
plt.ylabel('Mean Value')
plt.show()


# Save cluster descriptions
cluster_descriptions.to_csv('cluster_descriptions.csv')

# Save cluster means
cluster_means.to_csv('cluster_means.csv')
