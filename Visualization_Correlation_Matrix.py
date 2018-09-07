import plotly.plotly as py
from plotly.graph_objs import *
py.plotly.tools.set_credentials_file(username='lpegram', api_key='wc3S5UG8FWOPiq5ZgyAA')
import pylab
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import plotly.graph_objs as go

df = pd.read_csv("C:\\Users\Lisa.Pegram\\Downloads\\MPI_national.csv",skiprows = 1, header=None)
#url="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
#df = pd.read_csv(url,header=None)
print(df)
print(df.describe())

pd.options.display.max_columns=70


stats.probplot(df[4], dist="norm", plot=pylab)
pylab.show()

#The normal probability plot shows a non-linear pattern.
#The normal distribution is not a good model for these data.
print(df.corr)

pylab.show(df.corr()[0].plot())
pylab.show(df.corr().iloc[:,0:2].plot())

df2 = pd.DataFrame(pd.read_csv("C:\\Users\Lisa.Pegram\\Downloads\\MPI_national.csv"), columns=['MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban','MPI Rural', 'Headcount Ratio Rural',
           'Intensity of Deprivation Rural'])
print(df2.head(5))
###Check for categorial columns
#cols = df2.columns
#num_cols =df2._get_numeric_data()
#print(num_cols)
#print(list(set(cols) - set(num_cols)))
#print(df2.dtypes)

pd.plotting.scatter_matrix(df2, alpha=0.2, figsize=(12, 12),diagonal='kde')
plt.show()

print(df2.corr())
corr=df2.corr()

data = [
    go.Surface(
        z=corr.as_matrix()
    )
]
layout = go.Layout(
    title='3D-Visualization of the Correlation Matrix',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='correlations-as-a-3D-surface')
