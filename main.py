import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===============================
# 1) Load Dataset
# ===============================
df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# ===============================
# 2) Select Numeric Features
# ===============================
num_df = df.select_dtypes(include="number").copy()

# Remove CustomerID (it's just identifier)
if "CustomerID" in num_df.columns:
    num_df = num_df.drop(columns=["CustomerID"])

print("\nNumeric Columns Used:")
print(num_df.columns)

# ===============================
# 3) Handle Missing Values
# ===============================
print("\nMissing values:")
print(num_df.isna().sum())

# Fill with median (safe option)
num_df = num_df.fillna(num_df.median(numeric_only=True))

# ===============================
# 4) Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(num_df)

# ===============================
# 5) K-Means from k=2 to 10
# ===============================
ks = range(2, 11)
inertias = []
sil_scores = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# ===============================
# 6) Plot Elbow
# ===============================
plt.figure()
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.xticks(list(ks))
plt.show()

# ===============================
# 7) Plot Silhouette
# ===============================
plt.figure()
plt.plot(list(ks), sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.xticks(list(ks))
plt.show()

# ===============================
# 8) Choose Best k
# ===============================
best_index = np.argmax(sil_scores)
best_k = list(ks)[best_index]
best_score = sil_scores[best_index]

print("\nBest k based on Silhouette:", best_k)
print("Best Silhouette Score:", best_score)

# Final model
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

# Count customers per cluster
counts = pd.Series(final_labels).value_counts().sort_index()

print("\nCustomers per cluster:")
print(counts)