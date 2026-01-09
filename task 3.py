# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset (auto-detect separator)
file_path = "customer_segmentation.csv"  # Change path if needed

# Try tab separator first
try:
    df = pd.read_csv(file_path, sep="\t")
    if df.shape[1] == 1:  # If still single column, try comma
        df = pd.read_csv(file_path, sep=",")
except:
    df = pd.read_csv(file_path, sep=",")  # fallback

# Inspect data
print("First 5 rows:")
print(df.head())
print("\nColumns detected:")
print(df.columns)

# Select numeric columns automatically for clustering
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
print("\nNumeric columns to be used for clustering:", numeric_cols)

X = df[numeric_cols]

# Handle missing values (fill with mean)
X = X.fillna(X.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters with elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method to Determine Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with chosen clusters (change k based on elbow plot)
k = 4  # change based on elbow graph
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = clusters

# Display cluster info
print("\nNumber of customers in each cluster:")
print(df['Cluster'].value_counts())

print("\nSample clustered data:")
print(df.head())

# Visualize clusters using first two numeric features
if len(numeric_cols) >= 2:
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df[numeric_cols[0]],
        y=df[numeric_cols[1]],
        hue=df['Cluster'],
        palette='Set2',
        s=100
    )
    plt.title("Customer Segments")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.show()
