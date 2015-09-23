
# coding: utf-8

# In[15]:

# Used by organizers to calculate score


# In[16]:

import pandas as pd
import numpy as np


# In[18]:

for t in [('all_prediction', 'all_slope'), 
          ('test_prediction', 'test_slope'), 
         ]: 
    f = '../' + t[0] + ".csv"
    pred = pd.read_csv(f, sep = '|', index_col='SubjectID')
    actual = pd.read_csv('../' + t[1] + '.csv', sep = '|', index_col='SubjectID')
    j = pd.merge(pred, actual, left_index=True, right_index=True)
    print t, pred.shape, actual.shape, j.shape
    # The mean square error
    j.loc[:, 'SE'] = (j['prediction'] - j['ALSFRS_slope'])**2
    grouped_count = j.loc[:,['cluster', 'SE']].groupby('cluster').mean() 
    print t, grouped_count.apply(np.sqrt)    
    print t, "root mean square error: %.3f, size: %s" % (np.sqrt(np.mean(j['SE'])), j.shape)  
    print


# In[ ]:




# In[ ]:



