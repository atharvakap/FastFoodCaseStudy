import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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


