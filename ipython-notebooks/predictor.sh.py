
# coding: utf-8

# ## Run predictor.sh
# Read the challenge standard selected features and emit a prediction

# In[6]:

import pickle
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join
from StringIO import StringIO
from vectorizing_funcs import *

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../train_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../train_data_std.pickle', 'rb') )
model_per_cluster = pickle.load( open('../model_per_cluster.pickle', 'rb') )

if "IPython" not in sys.argv[0]:
    input_file, output_file= sys.argv[1], sys.argv[2]
else:
    input_file, output_file= "../selected_60879.txt", "../predicted_60879.txt"

def calc(x):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    pred = float(model.predict(x))
    return pd.Series({'prediction':pred, 'confidence': 0.5})
    
with open(input_file, 'r') as f:
    content = f.readlines()
    c = int(content[0].split(":")[1])
    s = "".join(content[1:])                 
    df = pd.read_csv(StringIO(s), sep='|', index_col=False,
                    names =["SubjectID","form_name","feature_name","feature_value","feature_unit","feature_delta"])
    feature_metadata = all_feature_metadata["Race"]
    vectorized, _ = vectorize(df, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    normalized.loc[:, "cluster"] = c
    pred = normalized.apply(calc, axis=1)
    print pred
    pred.to_csv(output_file % pred.index[0],sep='|', header=False, index=False, 
                columns=["prediction", "confidence"])


# In[ ]:



