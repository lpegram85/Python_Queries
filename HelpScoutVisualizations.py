import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
with open("C:\\Users\Lisa.Pegram\\Downloads\\Question5_.csv", 'r') as f:
    data2=pd.read_csv(f, sep=',')
    df = pd.DataFrame(data2.values)
print(df.head(5))

df=df.dropna(subset=[1,2])



B = df.iloc[:, 2]
print (B)
A = df.iloc[:, 1]
print (A)


print(pearsonr(A,B))
print(spearmanr(A,B))

A=np.array(A,dtype=float)
B=np.array(B,dtype=float)


results = sm.OLS(B,sm.add_constant(A)).fit()

print (results.summary())

plt.scatter(A,B,color='blue')

X_plot = np.linspace(0,20,900)
plt.plot(X_plot, X_plot*results.params[0] + results.params[1])
plt.xlabel('Mean Estimated User Group')
plt.ylabel('Rate of Growth')
plt.title('Scatter Matrix of Expected User v. Growth')
plt.show()