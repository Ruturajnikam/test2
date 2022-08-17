#!/usr/bin/env python
# coding: utf-8

# # Predicting student's dropout and academic success

# In[57]:


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[58]:


#reading dataset
df = pd.read_csv('C:/Users/rn21652/Desktop/data.csv') 


# In[59]:


df.head()


# In[60]:


df.info()


# In[61]:


#checking null values
df.isna().sum()


# In[62]:


df.describe()


# In[63]:


# Distribution graphs defining plot per column distribution
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[64]:


plotPerColumnDistribution(df, 37, 3)


# In[65]:


#correlation matrix
plt.figure(figsize=(37,37))
sns.heatmap(df.corr(),annot=True,cmap='viridis')


# # Splitting Train and Test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Target',axis=1), df['Target'], test_size=0.2, random_state=101)


# # Models

# In[67]:


#Random forest
model1 = RandomForestClassifier(n_estimators=200)
model1.fit(X_train,y_train)
predictions = model1.predict(X_test)
print("Accuracy: ", accuracy_score(y_test,predictions))


# In[68]:


print(classification_report(y_test,predictions))


# In[70]:


#logistic regression
model2 = LogisticRegression(max_iter=10000)
model2.fit(X_train,y_train)
predictions = model2.predict(X_test)
print("Accuracy: ", accuracy_score(y_test,predictions))


# In[71]:


print(classification_report(y_test,predictions))


# In[ ]:




