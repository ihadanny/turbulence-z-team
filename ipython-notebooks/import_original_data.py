
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np


# In[14]:

df = pd.read_csv('../../all_forms_PROACT.txt', sep = '|', error_bad_lines=False, index_col='SubjectID', dtype={'SubjectID': 'int'})
slope = pd.read_csv('../../ALSFRS_slope_PROACT.txt', sep = '|', index_col='SubjectID', 
                    dtype={'SubjectID': 'int', 'ALSFRS_slope': float})
slope = slope.dropna()
data_with_slope = pd.merge(df, slope, left_index=True, right_index=True)
print df.shape, slope.shape
print data_with_slope.shape, data_with_slope.index.unique().size
data_with_slope.head()


# In[16]:

data = data_with_slope.drop('ALSFRS_slope', 1)
data.to_csv('../all_data.csv',sep='|')
slope.to_csv('../all_slope.csv',sep='|')


# In[ ]:



