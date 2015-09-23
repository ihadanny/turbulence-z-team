
# coding: utf-8

# ## Run selector.sh
# As specified in the challenge - we must run our selector logic subject by subject.
# 
# The output_file_path must have the following format:
# * First line: the cluster identifier for that patient
# * Following lines: the selected features selected for that specific single patient, using the same format as the input data. A maximum of 6 features are allowed.

# In[4]:

import pickle
import cPickle
import pandas as pd
import sys
from vectorizing_funcs import *

if "IPython" not in sys.argv[0]:
    models_folder, input_file, output_file= sys.argv[1], sys.argv[2], sys.argv[3]
else:
    models_folder, input_file, output_file= "../", "../19871.txt", "../selected_19871.txt"

all_feature_metadata = pickle.load( open(models_folder + '/all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open(models_folder + '/all_data_means.pickle', 'rb') )
train_data_std = pickle.load( open(models_folder + '/all_data_std.pickle', 'rb') )
clustering_model = cPickle.load( open(models_folder + '/forest_clustering_model.pickle', 'rb') )
best_features_per_cluster = pickle.load( open(models_folder + '/best_features_per_cluster.pickle', 'rb') )
   
df = pd.read_csv(input_file, sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
for subj in df.SubjectID.unique()[:3]:
    df_subj = df[df.SubjectID == subj]
    vectorized, _ = vectorize(df_subj, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    c = np.digitize(clustering_model["model"].predict(normalized), clustering_model["bins"])[0]
    #cluster_data = normalized[clustering_model["columns"]]
    #c = clustering_model["model"].predict(cluster_data)[0]
    buf = "cluster: %d\n" % c
    selected = df_subj[df_subj.feature_name.isin(best_features_per_cluster[c])]
    buf += selected.to_csv(sep='|', index = False, header = False)
    print buf
    with open(output_file, "wb") as f:
        f.write(buf)


# In[ ]:



