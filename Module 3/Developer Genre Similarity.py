import pandas as pd
from scipy.sparse import lil_matrix
import scipy.spatial.distance

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
            for i in genre:
                if genre not in team_genre[team]:
                    team_genre[team][genre] = 0
                team_genre[team][genre] += 1 
            
# print(list(team_genre.items())[:2])

index = team_genre.keys()
rows = [team_genre[i] for i in index]

df = pd.DataFrame(rows, index = index)
df = df.fillna(0)
# print(df)
t1 = df.loc["Nintendo"]
t2 = df.loc["Electronic Arts"]
t3 = df.loc["Flying Wild Hog"]

distances1 = scipy.spatial.distance.cdist(df, [t1], metric = "euclidean").flatten()
distances2 = scipy.spatial.distance.cdist(df, [t2], metric = "euclidean").flatten()
distances3 = scipy.spatial.distance.cdist(df, [t3], metric = "euclidean").flatten()
# print(distances[:5])

query_distances1 = list(zip(df.index, distances1))
query_distances2 = list(zip(df.index, distances2))
query_distances3 = list(zip(df.index, distances3))
# print(query_distances[:5])

print("\nNINTENDO\n")
for similar_team, similar_genre_score, in sorted(query_distances1, key = lambda x:x[1])[:11]:
    print(similar_team, similar_genre_score)

print("\nELECTRONIC ARTS\n")
for similar_team, similar_genre_score, in sorted(query_distances2, key = lambda x:x[1])[:11]:
    print(similar_team, similar_genre_score)

print("\nFLYING WILD HOG\n")
for similar_team, similar_genre_score, in sorted(query_distances3, key = lambda x:x[1])[:11]:
    print(similar_team, similar_genre_score)
