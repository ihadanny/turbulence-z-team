
# coding: utf-8

# # Used for predicting ALSFRS_slope
# see https://www.synapse.org/#!Synapse:syn2873386/wiki/ .
# We assumed data is vectorized + clustered + 6 features were selected

# In[44]:

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from vectorizing_funcs import *


# ## Revectorize the selected data
# We now reload the metadata and the 6 attributes selected per cluster

# In[45]:

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../train_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../train_data_std.pickle', 'rb') )
best_features_per_cluster = pickle.load( open('../best_features_per_cluster.pickle', 'rb') )


df = pd.read_csv('../train_data_selected.csv', sep='|', index_col=False)
vectorized, _ = vectorize(df, all_feature_metadata)
normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
print normalized.shape
normalized.head()


# In[46]:

slope = pd.read_csv('../train_slope.csv', sep = '|', index_col="SubjectID")
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col="SubjectID")

X = normalized.join(clusters)
Y = slope.join(clusters)

print Y.shape, X.shape, clusters.shape


# ## Train a prediction model per cluster

# In[47]:

model_per_cluster = {}

for c in clusters.cluster.unique():    
    X_cluster = X[X.cluster==c]
    Y_cluster = Y[Y.cluster == c].ALSFRS_slope
    regr = linear_model.LinearRegression()
    regr.fit(X_cluster, Y_cluster)

    print 'cluster: %d size: %s' % (c, Y_cluster.shape)
    print "Mean square error (0 is perfect): %.2f" % np.mean(
        (regr.predict(X_cluster) - Y_cluster) ** 2)
    print('Explained variance score (1 is perfect): %.2f' % regr.score(X_cluster, Y_cluster))
    print ""
    model_per_cluster[c] = {"train_data_means": X_cluster.mean(), "model" : regr}
    
    


# ## Apply the model on both `train` and `test`

# In[58]:

def calc(x):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    pred = float(model.predict(x))
    return pd.Series({'SubjectID': int(x.name), 'prediction':pred, 'cluster': int(c), 'features_list': ";".join(best_features_per_cluster[c])})

for t in ['train', 'test']:
    df = pd.read_csv('../' + t + '_data_selected.csv', sep='|', index_col=False)
    vectorized, _ = vectorize(df, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col=0)
    X = normalized.join(clusters)
    pred = X.apply(calc, axis = 1)
    pred = pred.set_index('SubjectID')
    pred.to_csv('../' + t + '_prediction.csv',sep='|')

pred.head()


# ## Run predictor.sh
# Read the challenge standard selected features and emit a prediction

# In[59]:

import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from StringIO import StringIO
from IPython.display import display
from vectorizing_funcs import *

def calc(x):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    pred = float(model.predict(x))
    return pd.Series({'SubjectID': int(x.name), 'prediction':pred, 'cluster': int(c), 'features_list': ";".join(best_features_per_cluster[c])})

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../train_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../train_data_std.pickle', 'rb') )

for f_name in listdir('../'):
    if isfile(join('../',f_name)) and f_name.startswith('selected_'):
        with open(join('../',f_name), 'r') as f:
            content = f.readlines()
            c = int(content[0].split(":")[1])
            s = "".join(content[1:])                 
            df = pd.read_csv(StringIO(s), sep='|', index_col=False,
                            names =["SubjectID","form_name","feature_name","feature_value","feature_unit","feature_delta"])
            feature_metadata = all_feature_metadata["Race"]
            vectorized, _ = vectorize(df, all_feature_metadata)
            normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
            normalized.loc[:, "cluster"] = c
            display(normalized)
            pred = normalized.apply(calc, axis=1)
            display(pred)
            print pred.index[0]
            pred.to_csv('../predicted_%d.txt' % pred.index[0],sep='|')


# In[ ]:



