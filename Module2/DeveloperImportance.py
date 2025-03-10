import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 


g = nx.Graph()

df = pd.read_csv("games.csv")

for index, row in df.iterrows():
    teams = row["Team"]
    if isinstance(teams, str):
        team_list = [team.replace("'", "").replace("[", "").replace("]", "").strip() for team in teams.split(",")]  # Convert to string and split
        
    for team in team_list:
        g.add_node(team)
    
    for i in range(len(team_list)):
        for j in range(i + 1, len(team_list)):
            g.add_edge(team_list[i], team_list[j])
            
centrality_degree = nx.degree_centrality(g)
centrality_closeness = nx.closeness_centrality(g)
centrality_eigenvector = nx.eigenvector_centrality(g)

plt.figure(figsize =(200, 200)) 

nx.draw_networkx(g, with_labels = True) 
plt.show()

top_k = 10
print("DEGREE")
for u in sorted(centrality_degree, key=centrality_degree.get, reverse=True)[:top_k]:
    print(u, centrality_degree[u])
print("CLOSENESS")
for u in sorted(centrality_closeness, key=centrality_closeness.get, reverse=True)[:top_k]:
    print(u, centrality_closeness[u])
print("EIGENVECTOR")
for u in sorted(centrality_eigenvector, key=centrality_eigenvector.get, reverse=True)[:top_k]:
    print(u, centrality_eigenvector[u])
    

