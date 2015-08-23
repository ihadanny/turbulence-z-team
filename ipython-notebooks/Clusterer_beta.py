
# coding: utf-8

# In[1]:

## Used to create clusters of the vectorized data. Currently using good old k-means
## to visualize, we projected the data on 2d using PCA
## (taken from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)
## as the PCA decomposition showed dominance of Gender and Race, we took them out of the game before clustering 


# In[2]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.externals import joblib


plt.rcParams['figure.figsize'] = (15.0, 15.0)


# In[3]:

clustering_columns = [
    'ALSFRS_Total_last_zscore',
    'ALSFRS_Total_mean_slope_zscore',
    'weight_mean_zscore', 
    'weight_pct_diff_zscore',
    'Age_last_zscore',
    
    'onset_delta_last_zscore',
    'Albumin_last_zscore',
    'Creatinine_last_zscore',
    'fvc_percent_pct_diff_zscore',
    'bp_systolic_mean_zscore',
        
]


# In[4]:


###############################################################################
# Visualize results on PCA-reduced data
def visualize_kmeans(kmeans, data, resolution = 100):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Obtain labels for each point in mesh. Use last trained model.
#     Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure(1)
#     plt.clf()
#     plt.imshow(Z, interpolation='nearest',
#                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                cmap=plt.cm.Paired,
#                aspect='auto', origin='lower')

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, s=10, edgecolor='none', cmap='Paired')
    # Plot the centroids as a white X
    centroids = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                c=range(len(centroids)), zorder=10, cmap='Paired')
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.figure(figsize=(40,40))
    plt.show()



# In[5]:

# digits = load_digits()
# dig_data = scale(digits.data)

# n_samples, n_features = dig_data.shape
# n_digits = len(np.unique(digits.target))

# kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
# kmeans.fit(dig_data)
# visualize_kmeans(kmeans, dig_data)


# In[6]:

proact_train = pd.read_csv('../train_data_vectorized.csv', sep = '|', index_col = 'SubjectID', dtype='float')
proact_train = proact_train[clustering_columns]
proact_train.head()


# In[7]:

kmeans = KMeans(init='k-means++', n_clusters=3)
kmeans.fit(proact_train)
visualize_kmeans(kmeans, proact_train)
sorted([(metrics.adjusted_mutual_info_score(proact_train[col], kmeans.labels_), col) for col in proact_train.columns])
print "Cluster cnt: ", np.bincount(kmeans.labels_)


# In[8]:


for t in ['train', 'test']:
    cur_data = pd.read_csv('../' + t + '_data_vectorized.csv', sep = '|', error_bad_lines=False, index_col='SubjectID')
    cur_data = cur_data[clustering_columns]
    res = pd.DataFrame(index = cur_data.index)
    res['cluster'] = kmeans.predict(cur_data)
    res.to_csv('../' + t + '_kmeans_clusters.csv',sep='|')


# In[ ]:



