
# coding: utf-8

# In[10]:

# Used by organizers to calculate score


# In[11]:

import pandas as pd
import numpy as np


# In[16]:

for t in [('train_prediction', 'train_slope'), 
          ('test_prediction', 'test_slope'), 
          ('test_prediction_team_guy_zinman_raz_alon', 'test_slope'), 
          ('test_prediction_team_yanai', 'test_slope'), 
          ('test_prediction_team_zach', 'test_slope'), 
         ]: 
    f = '../' + t[0] + ".csv"
    pred = pd.read_csv(f, sep = '|', index_col='SubjectID')
    actual = pd.read_csv('../' + t[1] + '.csv', sep = '|', index_col='SubjectID')
    j = pd.merge(pred, actual, left_index=True, right_index=True)
    print t, pred.shape, actual.shape, j.shape
    # The mean square error
    print f + " mean square error: %.2f, size: %s" % (np.mean((j['prediction'] - j['ALSFRS_slope']) ** 2), j.shape)  


# In[ ]:



