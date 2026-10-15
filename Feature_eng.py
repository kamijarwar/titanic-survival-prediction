#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
from sklearn .model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.head(10)


# In[4]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# In[5]:


df.head(3)


# In[6]:


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[7]:


df.info()


# In[8]:


df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['ISalone'] = (df['Family_Size'] == 1).astype(int)
df.info()


# In[9]:


df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# In[10]:


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
cols = ['Sex_male', 'Embarked_Q', 'Embarked_S']
df[cols] = df[cols].astype(int)


# In[11]:


df.info()


# In[12]:


df.head()


# In[13]:


df.drop('Cabin', axis=1, inplace=True)


# In[14]:


df.columns


# In[15]:


X = df.drop('Survived', axis = 1)
y = df['Survived']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state= 42)


# In[17]:


print (X_train.shape, X_test.shape)


# In[20]:


scaler = StandardScaler()
cols = ['Age', 'Fare', 'Family_Size']


# In[21]:


X_train[cols] = scaler.fit_transform(X_train[cols])


# In[22]:


X_test[cols] = scaler.transform(X_test[cols])


# In[24]:


model = LogisticRegression(max_iter = 1000)
model.fit(X_train,y_train)


# In[25]:


y_pred = model.predict(X_test)


# In[28]:


print ("Accuracy :",accuracy_score(y_test, y_pred))


# In[31]:


cols_drop = ['Family_Size', 'ISalone']


# In[32]:


X_train_old = X_train.drop(cols_drop, axis=1)
X_test_old = X_test.drop(cols_drop, axis=1)


# In[33]:


model_old = LogisticRegression(max_iter=1000)
model_old.fit(X_train_old, y_train)


# In[37]:


print("without new  features:", accuracy_score(y_test, model_old.predict(X_test_old)))


# In[38]:


print("With New feature:", accuracy_score(y_test, model.predict(X_test)))


# In[ ]:




