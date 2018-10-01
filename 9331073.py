
# coding: utf-8

# In[21]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

pd.options.display.max_rows = 800


# In[22]:


df_train = pd.read_csv('train.csv')


# In[23]:


X_train = df_train.values[:, 0:9]
Y_train = df_train.values[:,9]


# In[24]:


df_test = pd.read_csv('test.csv')


# In[25]:


df_test['Steel'].fillna('A', inplace=True)


# In[26]:


df_test['Condition'].fillna('S', inplace=True)


# In[27]:


mean_value= df_test['Formability'].mean()
df_test['Formability']=df_test['Formability'].fillna(mean_value)


# In[28]:


df_test.loc[df_test['Surface_Quality'].isnull() & (df_test['Steel'] == "R"), 'Surface_Quality'] = df_test.loc[df_test['Surface_Quality'].isnull() & (df_test['Steel'] == "R"), 'Surface_Quality'].fillna('E')


# In[29]:


df_test.loc[df_test['Surface_Quality'].isnull() & (df_test['Steel'] == "A") & (df_test[' shape'] == "SHEET"), 'Surface_Quality'] = df_test.loc[df_test['Surface_Quality'].isnull() & (df_test['Steel'] == "A") & (df_test[' shape'] == "SHEET"), 'Surface_Quality'].fillna('F')


# In[30]:


df_test.loc[df_test['Surface_Quality'].isnull() & (df_test['Formability'] > 2) & (df_test['Formability'] < 3), 'Surface_Quality'] = df_test.loc[df_test['Surface_Quality'].isnull() & (df_test['Formability'] > 2) & (df_test['Formability'] < 3), 'Surface_Quality'].fillna('G')


# In[31]:


df_test['Surface_Quality'].fillna('E', inplace=True)


# In[32]:


mean_value= df_test['Formability'].mode()
df_test['Formability']=df_test['Formability'].fillna(mean_value)


# In[33]:


clean1 = {"Steel": {"R": 0, "A":1, "U":2, "K":3, "M":4, "S":5, "W":6, "V":7}}
df_test.replace(clean1, inplace=True)
clean2 = {" shape": {"COIL": 0, "SHEET":1}}
df_test.replace(clean2, inplace=True)
clean3 = {"Condition": {"S":0, "A":1, "X":2}}
df_test.replace(clean3, inplace=True)
clean4 = {"Surface_Quality": {"D": 0, "E":1, "F":2, "G":3}}
df_test.replace(clean4, inplace=True)


# In[34]:


X_test = df_test.values[:, 0:9]


# In[35]:


clf_gini = DecisionTreeClassifier(criterion = "entropy", max_depth=11, min_samples_split=2)
clf_gini.fit(X_train, Y_train)
y_pred = clf_gini.predict(X_test)


# In[36]:


np.savetxt('results.csv', y_pred, delimiter=" ", fmt="%s")

