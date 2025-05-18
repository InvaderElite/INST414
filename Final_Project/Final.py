import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("sleep_cycle_productivity.csv")
# Module 1

top_common_sleep_hours = df["Total Sleep Hours"].round(1).value_counts().nlargest(10).sort_index()

# Plot the top 10 most common sleep durations
plt.figure(figsize=(10, 6))
sns.barplot(x=top_common_sleep_hours.index, y=top_common_sleep_hours.values, palette="coolwarm")
plt.title("Top 10 Most Common Total Sleep Hours")
plt.xlabel("Total Sleep Hours (rounded to nearest 0.1 hr)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


heatmap_data = df.copy()
heatmap_data["Sleep Hours Bin"] = heatmap_data["Total Sleep Hours"].round(0)
heatmap_data["Sleep Quality Bin"] = heatmap_data["Sleep Quality"]

# Create a pivot table of counts
pivot_table = heatmap_data.pivot_table(
    index="Sleep Quality Bin", 
    columns="Sleep Hours Bin", 
    aggfunc="size", 
    fill_value=0
)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Heatmap of Total Sleep Hours vs Sleep Quality")
plt.xlabel("Total Sleep Hours (rounded to nearest hr)")
plt.ylabel("Sleep Quality")
plt.tight_layout()
plt.show()

# Module 4

data = df[['Sleep Start Time', 'Sleep End Time']].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(data)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Sleep Start Time', y='Sleep End Time', hue='Cluster', palette='viridis', s=60)
plt.xticks(range(19, 25)) 
plt.yticks(range(1, 10))
plt.title('Clusters of Sleep Start and End Times')
plt.xlabel('Sleep Start Time')
plt.ylabel('Sleep End Time')
plt.grid(True)
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


# Overlays
plot_data = df[['Sleep Start Time', 'Sleep End Time', 'Age']].dropna()

plt.figure(figsize=(12, 7))
scatter = plt.scatter(
    plot_data['Sleep Start Time'],
    plot_data['Sleep End Time'],
    c=plot_data['Age'],
    cmap='plasma',
    s=30,
    alpha=0.7
)

cbar = plt.colorbar(scatter)
cbar.set_label('Age')

plt.title('Sleep Start vs End Time Colored by Age')
plt.xlabel('Sleep Start Time')
plt.ylabel('Sleep End Time')
plt.xticks(range(19, 25))  
plt.yticks(range(0, 10)) 
plt.grid(True)
plt.tight_layout()
plt.show()


plot_data = df[['Sleep Start Time', 'Sleep End Time', 'Sleep Quality']].dropna()

plt.figure(figsize=(12, 7))
scatter = plt.scatter(
    plot_data['Sleep Start Time'],
    plot_data['Sleep End Time'],
    c=plot_data['Sleep Quality'],
    cmap='viridis',
    s=30,
    alpha=0.7
)

cbar = plt.colorbar(scatter)
cbar.set_label('Sleep Quality')

plt.title('Sleep Start vs End Time Colored by Sleep Quality')
plt.xlabel('Sleep Start Time')
plt.ylabel('Sleep End Time')
plt.xticks(range(19, 25))  
plt.yticks(range(0, 10))  
plt.grid(True)
plt.tight_layout()
plt.show()


plot_data = df[['Sleep Start Time', 'Sleep End Time', 'Mood Score']].dropna()

plt.figure(figsize=(12, 7))
scatter = plt.scatter(
    plot_data['Sleep Start Time'],
    plot_data['Sleep End Time'],
    c=plot_data['Mood Score'],
    cmap='cividis',
    s=30,
    alpha=0.7
)

cbar = plt.colorbar(scatter)
cbar.set_label('Mood Score')

plt.title('Sleep Start vs End Time Colored by Mood Score')
plt.xlabel('Sleep Start Time')
plt.ylabel('Sleep End Time')
plt.xticks(range(19, 25)) 
plt.yticks(range(0, 10))   
plt.grid(True)
plt.tight_layout()
plt.show()

