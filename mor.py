import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.mixture import GaussianMixture
from patsy import dmatrices
from flexmix import flexmix, step, refit
import matplotlib.pyplot as plt

mcdonalds = np.read_csv('mcdonalds.csv')
counts = pd.value_counts(mcdonalds["Like"])
reverse_counts = counts.iloc[::-1]

mcdonalds["Like.n"] = 6 - pd.to_numeric(mcdonalds["Like"])
counts = pd.value_counts(mcdonalds["Like.n"])

f = "+".join(mcdonalds.columns[0:11])
f = "Like.n ~ " + f
f = smf.ols(formula=f)

formula_str = "Like.n ~ yummy + convenient + spicy + fattening + greasy + fast + cheap + tasty + expensive + healthy + disgusting"
formula_obj = smf.ols(formula=formula_str)

np.random.seed(1234)
y, X = dmatrices(str(f), data=mcdonalds, return_type='dataframe')
gmm = GaussianMixture(n_components=2, n_init=10)
gmm.fit(X)
md_reg2 = flexmix(y.values.ravel(), X.values, k=2, cluster=gmm.predict(X))

y, X = dmatrices(str(f), data=mcdonalds, return_type='dataframe')
np.random.seed(1234)
model = GaussianMixture(n_components=2, n_init=10)
search = step(model, X.values, y.values.ravel(), k=2, nrep=10, verb=0)
md_reg2 = flexmix(y.values.ravel(), X.values, k=search['nclass'], cluster=search['class'])

y, X = dmatrices(str(f), data=mcdonalds, return_type='dataframe')
np.random.seed(1234)
model = GaussianMixture(n_components=2, n_init=10)
search = step(model, X.values, y.values.ravel(), k=2, nrep=10, verb=0)
md_reg2 = flexmix(y.values.ravel(), X.values, k=search['nclass'], cluster=search['class'])
md_ref2 = refit(md_reg2)
print(md_ref2.summary())

y, X = dmatrices(str(f), data=mcdonalds, return_type='dataframe')
np.random.seed(1234)
model = GaussianMixture(n_components=2, n_init=10)
search = step(model, X.values, y.values.ravel(), k=2, nrep=10, verb=0)
md_reg2 = flexmix(y.values.ravel(), X.values, k=search['nclass'], cluster=search['class'])
md_ref2 = refit(md_reg2)

# plot the results
plt.figure()
plt.scatter(X['Like.n'], y, c=md_ref2.predict(X.values))
plt.colorbar()
plt.xlabel('Like.n')
plt.ylabel('Like')
plt.title('Cluster plot of McDonald\'s data')
plt.show()
