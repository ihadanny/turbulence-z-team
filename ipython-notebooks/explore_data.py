
# coding: utf-8

# In[56]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
plt.rcParams['figure.figsize'] = (10.0, 10.0)

from  vectorizing_funcs import *
df = pd.read_csv('../all_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.head()


# In[81]:

good_features = [
                'ALSFRS_Total', 'weight', 'Albumin', 'Creatinine',
            'bp_diastolic', 'bp_systolic', 'pulse', 'respiratory_rate', 'temperature',
            'mouth', 'respiratory', 'hands', 'fvc_percent'
            'BMI', 'Age', 'onset_delta',
            'family_ALS_hist', 'if_use_Riluzole',
'Albumin',
        'Alkaline Phosphatase',
        'ALPHA1-GLOBULIN',
        'ALPHA2-GLOBULIN',
        'ALT(SGPT)',
        'Amylase',
        'AST(SGOT)',
        'Bilirubin (Direct)',
        'Bilirubin (Indirect)',
        'Bilirubin (Total)',
        'Blood Urea Nitrogen (BUN)',
        'Calcium',
        'Chloride',
        'CK',
        'C-Reactive Protein',
        'Creatine Kinase MB',
        'Creatinine',
        'Erythrocyte Sediment',
        'Ferritin',
        'Fibrinogen',
        'Free T3',
        'Free T4',
        'Free Thyroxine Index',
        'GAMMA-GLOBULIN',
        'Gamma-glutamyltransferase',
        'Glucose',
        'HbA1c (Glycated Hemoglobin)',
        'HDL',
        'Hemoglobin',
        'Lactate Dehydrogenase',
        'LDL',
        'Lymphocytes',
        'Magnesium',
        'Mean Corpuscular Hemoglobin',
        'Mean Platelet Volume',
        'Monocytes',
        'Neutrophils',
        'Parathyroid Hormone',
        'Phosphorus',
        'Platelets',
        'Potassium',
        'Protein',
        'RBC Morphology: Spherocytes',
        'RBC Morphology: Target Cells',
        'RBC Morphology: Tear drop cells',
        'Sodium',
        'Thyroid Stimulating Hormone',
        'Total Cholesterol',
        'Total T4',
        'Triglycerides',
        'Uric Acid',
        'Urine Albumin',
        'Vitamin B12',
        'White Blood Cell (WBC)'
]


# In[82]:

a = df[df.feature_name.isin(good_features)]
a.loc[:, 'subj_time'] = a.SubjectID + "_" + a.feature_delta.convert_objects(convert_numeric=True).astype(str)
a = a.drop_duplicates(subset = ['subj_time', 'feature_name'], take_last=True)
a.loc[:, 'feature_value'] = a.feature_value.convert_objects(convert_numeric=True)
p = a.pivot(index='subj_time', columns='feature_name', values='feature_value')
display(p.head())
print p.shape


# In[40]:

distrib_per_col = a.groupby('feature_name')['feature_value'].agg(['mean', 'std'])
j = pd.merge(a, distrib_per_col, left_on='feature_name', right_index=True)
j.loc[:, 'critical'] = abs(j['mean']) + j['std']*3
outliers = j[abs(j.feature_value) > abs(j.critical)]


# In[43]:

display(outliers.head())
outliers.to_csv('../outliers.txt', sep = '|', index=False)


# In[128]:

import scipy
for col in good_features:
    b = df[df.feature_name == col]
    b_num = b.feature_value.convert_objects(convert_numeric=True)
    if b_num.dtypes == np.float64 or b_num.dtypes == np.int64:
        b_num = b_num[~np.isnan(b_num)]
        if b_num.size > 10:
            sns.distplot(b_num, axlabel = col)
            plt.show()
            medi = np.median(b_num)
            mad = np.median(abs(b_num - medi))
            print col, "median:", medi, "mad:", mad
            if mad > 0.0:
                sns.distplot(b_num[abs(b_num-medi) < 4 * mad], axlabel = col)
                plt.show()
            
    


# In[5]:

feature_names = df[["form_name", "feature_name"]].drop_duplicates()
feature_names.to_csv('../feature_names.csv', sep='|', index=False)


# In[3]:

feature_values = df[["form_name", "feature_name", "feature_value"]].drop_duplicates()
feature_values = feature_values[np.isnan(feature_values.feature_value.convert_objects(convert_numeric=True))]
feature_values.to_csv('../feature_values.csv', sep='|', index=False)


# In[5]:

feature_values = df[df.form_name == 'Lab Test']
feature_values = feature_values[~np.isnan(feature_values.feature_value.convert_objects(convert_numeric=True))]
by_subject = feature_values.groupby(["feature_name", "SubjectID"])
features_with_multiple_visits = by_subject.filter(lambda x: len(x)>2)
by_subject = features_with_multiple_visits.groupby("feature_name").SubjectID.nunique()
by_subject.sort(ascending=False)
by_subject[:30]


# In[100]:

all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))
all_feature_metadata = learn_to_dummies_model(df, all_feature_metadata)
vectorized, all_feature_metadata = vectorize(df, all_feature_metadata, debug=True)


# In[101]:

vectorized.describe().transpose()


# In[30]:

slope = pd.read_csv('../all_slope.csv', sep = '|', index_col=0)
slope.index = slope.index.astype(str)

max_date = df[df.feature_name == 'ALSFRS_Total'][['SubjectID','feature_delta']]
max_date.loc[:, 'feature_delta'] = max_date.feature_delta.astype(int)
max_date = max_date.groupby('SubjectID').max()
print max_date.shape, slope.shape
j = slope.join(max_date)
print j.shape
j[j.feature_delta < 365].shape


# In[33]:

df[df.feature_name == 'onset_site'].feature_value.unique()


# In[7]:

slope = pd.read_csv('../all_slope.csv', sep = '|', index_col=0)
filtered = pd.read_csv('../../ALSFRS_slope_PROACT_filtered.txt', sep = '|', index_col=0)
filtered.head()


# In[8]:

slope.head()


# In[19]:

j = pd.merge(filtered, slope, how='right', left_index = True, right_index = True)


# In[22]:

j[np.isnan(j.ALSFRS_slope_x)].head()


# In[ ]:



