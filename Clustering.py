import matplotlib.pyplot as plt
#%matplotlib inline
import plotly.plotly as py
import plotly.graph_objs as go
py.plotly.tools.set_credentials_file(username='lpegram', api_key='wc3S5UG8FWOPiq5ZgyAA')
import pandas as pd # should have brought data in as a data frame ignoring the headers
import io
import numpy as np
from numpy.polynomial.polynomial import polyfit
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.mixture import GMM
from matplotlib.patches import Ellipse

plt.interactive(False)
# create an array of random variables
np.random.seed(2)


#BE LOANS Conventional
f1=np.genfromtxt("C:\\Users\Lisa.Pegram\\Downloads\\CreditScore.csv", delimiter=","
                               ,skip_header=1)
f2=np.genfromtxt("C:\\Users\Lisa.Pegram\\Downloads\\ConvBEMargin.csv", delimiter=","
                               ,skip_header=1)

dataforanalysis=np.array(list(zip(f1,f2)))


#print(dataforanalysis)

plt.show(plt.scatter(dataforanalysis[:, 0], dataforanalysis[:, 1],c='black', s=7))

#original= go.Scatter(
#    x = f1,
#    y = f2,
#    mode = 'markers')
#data=[original]

#py.iplot(data,filename='Scatter for Cluster')

kmeans=KMeans(n_clusters=3)
kmeans.fit(dataforanalysis)
kmeans = kmeans.fit(dataforanalysis)
labels= kmeans.predict(dataforanalysis)
#print(y_kmeans)

centers = kmeans.cluster_centers_
print(centers)
b, m = polyfit(f1,f2,1)

fig=plt.figure()
ax=fig.add_subplot(111)

ax.scatter(dataforanalysis[:, 0], dataforanalysis[:, 1], c=labels)
ax.plot(f1, b + m *f1 , '-')
ax.scatter(centers[:, 0], centers[:, 1],marker='*', c='#050505', s=20)
plt.show()


#Gaussian Mixsture Models


gmm = GMM(n_components=3).fit(dataforanalysis)
labels = gmm.predict(dataforanalysis)
plt.show(plt.scatter(dataforanalysis[:, 0], dataforanalysis[:, 1], c=labels, s=10, cmap='viridis'))


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

#this is calculating the dot product to do analysis of clustering with

rng = np.random.RandomState(2)
dataforanalysis_stretched = np.dot(dataforanalysis, rng.randn(2, 951))

#gmm = GMM(n_components=3, covariance_type='full', random_state=36)
#plot_gmm(gmm,dataforanalysis_stretched)

##print(kmeans.labels_)

#print(type(dataforanalysis))

###########plt.show(plt.scatter(dataforanalysis[:,0],dataforanalysis[:,1], c=kmeans.labels_, s=200, alpha=0.5))


def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(dataforanalysis, 3)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataforanalysis[:, 0], dataforanalysis[:, 1], c=labels,s=50, cmap='viridis')
ax.scatter(centers[:, 0], centers[:, 1],marker='*', c='#060505', s=100)
plt.show()

#Silhoutte Analysis

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(dataforanalysis) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(dataforanalysis)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dataforanalysis, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dataforanalysis, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(dataforanalysis[:, 0], dataforanalysis[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()