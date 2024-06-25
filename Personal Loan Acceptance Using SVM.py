#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Necessary Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('UniversalBank.csv') # reading the data
df


# In[3]:


# Checking for null values
df.isnull().sum()


# In[4]:


# Dropping ID and ZIP Code columns from the dataset
df1 = df.drop(["ID","ZIP Code"], axis = 1)
df1.head()


# In[5]:


#Using Heatmap to show correlation between all features
plt.figure(figsize=(15,8))
plt.title("Heatmap showing Correlation between all the features", fontsize=20)
sns.heatmap(df1.corr(),annot = True, cmap='mako')


# In[6]:


#identifying non-credit card holders 
zero_class = df1[df1.CreditCard==0]
zero_class.shape #(row,columns)


# In[7]:


#identifying credit card holders 
one_class = df1[df1.CreditCard==1]
one_class.shape #(row,columns)


# In[8]:


# Income vs Experience scatter plot
plt.xlabel('Income')
plt.ylabel('Experience')
plt.title("People Not-Having Credit Card")
plt.scatter(zero_class['Income'],zero_class['Experience'], color = 'green', marker='.')


# In[9]:


# Income vs Experience scatter plot
plt.xlabel('Income')
plt.ylabel('Experience')
plt.title("People Having Credit Card")
plt.scatter(one_class['Income'], one_class['Experience'], color = 'red', marker='.')


# In[10]:


# CCAvg vs Family scatter plot
plt.xlabel('CCAvg')
plt.ylabel('Family')
plt.scatter(zero_class['CCAvg'],zero_class['Family'], color = 'blue', marker='+')
plt.scatter(one_class['CCAvg'], one_class['Family'], color = 'red', marker='.')


# In[11]:


# Scaling the data using Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit(df1.drop('CreditCard',axis=1)).transform(df1.drop('CreditCard',axis=1))
df_scaled = pd.DataFrame(scaled, columns=df1.columns[:-1])
df_scaled.head()


# In[12]:


# Splitting the columns in to dependent variable (x) and independent variable (y).
x = df_scaled
y = df1['CreditCard']


# In[ ]:





# In[13]:


# Split data in to train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)


# In[14]:


# Apply SVM Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc=SVC() 
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
print('Model accuracy : {0:0.3f}'. format(accuracy_score(y_test, y_pred)))


# In[15]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')


# In[16]:


# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




