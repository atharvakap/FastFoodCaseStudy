import numpy as np
import pandas as pd
from flexmix import flexmix, getModel
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(1234)

data = pd.read_csv("mcdonalds.csv")

k_range = range(2, 9)
n_reps = 10

MD_m28 = flexmix(formula="MD.x ~ 1", data=data, k=k_range, nrep=n_reps, model="FLXMCmvbinary", control=list(verbose=FALSE))

print(MD_m28)

plt.plot(MD_m28)
plt.ylabel("value of information criteria (AIC, BIC, ICL)")
plt.show()

MD_m4 = getModel(MD_m28, which="4")

kmeans_clusters = flexmix_result_model["cluster"]
mixture_clusters = MD_m4.predict()

contingency_table = pd.crosstab(index=kmeans_clusters, columns=mixture_clusters, rownames=["kmeans"], colnames=["mixture"])

print(contingency_table)


kmeans_clusters = flexmix_result_model["cluster"]

MD_m4a = flexmix(formula="MD.x ~ 1", data=data, cluster=kmeans_clusters, model="FLXMCmvbinary")

kmeans_clusters = flexmix_result_model["cluster"]
mixture_clusters = MD_m4a.predict()

contingency_table = pd.crosstab(index=kmeans_clusters, columns=mixture_clusters, rownames=["kmeans"], colnames=["mixture"])

print(contingency_table)

loglik_m4a = np.sum(np.log(np.sum(MD_m4a.weights * np.array([multivariate_normal.pdf(data, mean=mean, cov=cov, allow_singular=True) for mean, cov in zip(MD_m4a.params['mean'], MD_m4a.params['Sigma'])]), axis=1)))
print(f"log Lik. {loglik_m4a:.3f} (df={MD_m4a.nparam})")

loglik_m4 = np.sum(np.log(np.sum(MD_m4.weights * np.array([multivariate_normal.pdf(data, mean=mean, cov=cov, allow_singular=True) for mean, cov in zip(MD_m4.params['mean'], MD_m4.params['Sigma'])]), axis=1)))
print(f"log Lik. {loglik_m4:.3f} (df={MD_m4.nparam})")

