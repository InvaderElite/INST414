import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

team_genre = {}

df = pd.read_csv("games.csv")

for index, row in df.iterrows():
    teams = row["Team"]
    genres = row["Genres"]
    if isinstance(teams, str):
        team_list = [team.replace("'", "").replace("[", "").replace("]", "").strip() for team in teams.split(",")]  # Convert to string and split
    
    if isinstance(genres, str):
        genre_list = [genre.replace("'", "").replace("[", "").replace("]", "").strip() for genre in genres.split(",")]
        
    for team in team_list:
        if team not in team_genre:
            team_genre[team] = {}
        for genre in genre_list:
            if genre not in team_genre[team]:
                team_genre[team][genre] = 0
            team_genre[team][genre] += 1 

index = team_genre.keys()
rows = [team_genre[i] for i in index]

df = pd.DataFrame(rows, index = index)
df = df.fillna(0)

# print(df[:2])

inertia_values = []
for k in range(2,31):
    kmeans = KMeans(n_clusters = k)
    
    kmeans.fit(df)
    
    inertia_values.append(kmeans.inertia_)
# print(inertia_values)

prev = inertia_values[0]
for inertia in inertia_values[1:]:
    # print(prev - inertia)
    prev = inertia
    
plt.figure(figsize = (15,6))
plt.title("KMeans")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS Values")
plt.plot(range(2, 31), inertia_values, marker = "o") #can use plt.xticks(range(2, 31)) for better visualization
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters = 8)

cluster_labels = kmeans.fit_predict(df)
# print(cluster_labels[:5])

df["cluster"] = cluster_labels
# print(df[:5])

grouped = df.groupby("cluster")
# print(list(grouped)[:5])

print("Number of actors per cluster")
print(df["cluster"].value_counts())

for cluster, group in grouped:
    print(f"\nCluster {cluster}:")
    try:
        sample_devs = group.sample(n = 5).index
        for dev in sample_devs:
            print(dev)
    except:
        sample_devs = group.sample(n = 1).index
        for dev in sample_devs:
            print(dev)

features = df.drop(columns=["cluster"])

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)


plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=df["cluster"], cmap="tab10", s=60, alpha=0.7)

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("Game Dev Team Clusters by Genre")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

pca_components = pd.DataFrame(pca.components_, columns=features.columns, index=['PC1', 'PC2'])

for pc in pca_components.index:
    print(f"\nTop genres contributing to {pc}:")
    print(pca_components.loc[pc].sort_values(ascending=False).head(4))
    print(pca_components.loc[pc].sort_values(ascending=False).tail(4))
