import pandas as pd
import kagglehub
import matplotlib.pyplot as plt


# Download latest version
path = kagglehub.dataset_download("arnabchaki/popular-video-games-1980-2023")

print("Path to dataset files:", path)

url = 'games.csv'
games = pd.read_csv(url)

df = pd.DataFrame(games)
# print(df[["Title", "Team", "Rating", "Genres"]])

genre_dict = {}

# Iterate through the dataset
for index, row in df.iterrows():
    genres = str(row["Genres"]).replace("'", "").replace("[", "").replace("]", "").split(",")  # Convert to string and split
    companies = str(row["Team"]).replace("'", "").replace("[", "").replace("]", "").split(",")  # Convert to string and split
        
        
    genres = [genre.strip() for genre in genres if genre.strip()]
    companies = [company.strip() for company in companies if company.strip()]
    
    for genre in genres:
        if genre not in genre_dict or None:
            genre_dict[genre] = set()  # Use a set to avoid duplicates
        genre_dict[genre].update(companies)  # Add all companies to the set

genre_dict = {genre: list(companies) for genre, companies in genre_dict.items()}

data = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in genre_dict.items()]))
# data = data.fillna("")
# print(data)

genre_count = []
comp_count = []
for genre in data:
    genre_count.append(genre)
    comp_count.append(data[genre].count())
# print(game_count)

game_stats = {
    "genres": genre_count,
    "companies": comp_count
}
genre_stats = pd.DataFrame(game_stats)
print(genre_stats)

genre_stats["companies"].plot(kind = "barh")  
plt.yticks(ticks=range(len(genre_stats.index)), labels=genre_stats["genres"].values)

# plt.show()