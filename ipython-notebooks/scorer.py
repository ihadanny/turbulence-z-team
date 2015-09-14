
# coding: utf-8

# In[1]:

# Used by organizers to calculate score


# In[2]:

import pandas as pd
import numpy as np


# In[5]:

for t in [('all_prediction', 'all_slope'), 
          ('test_prediction', 'test_slope'), 
         ]: 
    f = '../' + t[0] + ".csv"
    pred = pd.read_csv(f, sep = '|', index_col='SubjectID')
    actual = pd.read_csv('../' + t[1] + '.csv', sep = '|', index_col='SubjectID')
    j = pd.merge(pred, actual, left_index=True, right_index=True)
    print t, pred.shape, actual.shape, j.shape
    # The mean square error
    print f + " mean square error: %.3f, size: %s" % (np.mean((j['prediction'] - j['ALSFRS_slope']) ** 2), j.shape)  
    print


# In[ ]:




# In[ ]:



