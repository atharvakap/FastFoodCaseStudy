import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from flexclust import stepFlexclust, relabel
import matplotlib.pyplot as plt

np.random.seed(1234)
MD = pd.read_csv('mcdonalds.csv')
# Assuming MD.x is a numpy array or pandas DataFrame containing the data
MD_kms = stepFlexclust(MD.x, k=range(2, 9), nrep=10, verbose=False, method='kmeans')

# Relabel the clusters to ensure consistency across runs
MD_kms = relabel(MD_kms)

plt.plot(range(2, 9), MD_kms["BIC"], "bo-", label="BIC")
plt.xlabel("number of segments")
plt.legend()
plt.show()

MD_b28 = bootFlexclust(MD.x, k=range(2,9), nrep=10, nboot=100, method='kmeans')

plt.plot(range(2, 9), MD_b28["ARI"], "bo-", label="Adjusted Rand Index")
plt.xlabel("number of segments")
plt.ylabel("adjusted Rand index")
plt.legend()
plt.show()

cluster_data = MD.x[MD_kms["cluster_id"] == "4", :]
plt.hist(cluster_data, range=[0,1])
plt.xlim(0,1)
plt.xlabel("Variable values")
plt.ylabel("Frequency")
plt.show()

MD_k4 = MD_kms["4"]
MD_r4 = slswFlexclust(MD.x, MD_k4)

plt.plot(MD_r4["ss"], "bo-", label="segment stability")
plt.ylim(0,1)
plt.xlabel("segment number")
plt.ylabel("segment stability")
plt.legend()
plt.show()
