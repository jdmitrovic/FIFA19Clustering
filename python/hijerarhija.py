est = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', affinity='euclidean')
est.fit(scaled_df)
df['labels'] = est.labels_

print("Silhouette score: %f " % silhouette_score(scaled_df, est.labels_))