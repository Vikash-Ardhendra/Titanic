#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


os.getcwd()
os.chdir('C:/Users/DELL/OneDrive/Desktop/DS Project/titanic')
train = pd.read_csv('train.csv')
test = pd.read_csv ('test.csv')
gender = pd.read_csv('gender_submission.csv')


# In[3]:


test= pd.merge(test,gender,how='outer')


# In[4]:


test


# In[5]:


df = pd.merge(train,test,how='outer')
df


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# In[8]:


df= df.drop(columns=['Ticket','Cabin'])


# In[9]:


df


# In[10]:


from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='median')
df['Age']=si.fit_transform(df[['Age']])
df['Fare']=si.fit_transform(df[['Fare']])


# In[11]:


si2= SimpleImputer(strategy='most_frequent')
df['Embarked']=si2.fit_transform(df[['Embarked']])
df


# In[12]:


df['Embarked']=df['Embarked'].astype('category').cat.codes
df['Sex']=df['Sex'].astype('category').cat.codes


# In[13]:


df


# ###  Exploratory Analysis

# In[14]:


sns.catplot(data=train,x='Sex',hue='Survived',kind='count')
plt.show()


# In[15]:


train_x = df.drop(columns=['SibSp','Parch','Fare','Name','PassengerId','Survived'])
train_x


# In[16]:


y = df.pop('Survived')
train_y = pd.DataFrame(y,columns=['Survived'])
train_y


# In[17]:


train_x = pd.get_dummies(train_x,columns=['Pclass','Embarked'])
train_x


# In[18]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
train_x = std.fit_transform(train_x)


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size=0.25, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[20]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ### Logistic Regression

# In[21]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(x_train,y_train)
lr_test = lr.predict (x_test)
lr_acc = accuracy_score (y_test , lr_test) * 100
lr_acc


# In[22]:


lr_test1 = lr.predict (x_train)
lr_acc1 = accuracy_score (y_train , lr_test1) * 100
lr_acc1


# ###  Accuracy Score of Logistic Regression is 85.6

# ###  Decision Tree

# In[23]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(criterion='entropy')
dc = dc.fit( x_train, y_train)
dc_pred = dc.predict(x_test)
dc_acc = accuracy_score(y_test, dc_pred)
dc_acc


# In[24]:


dc_test1 = dc.predict (x_train)
dc_acc1 = accuracy_score (y_train , dc_test1) * 100
dc_acc1


# ### Decision Tree 
# Training Score = 91
# Test Score = 83

# ###  Randomn Forest

# In[25]:


from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(220,criterion='entropy',random_state=42)
rc = rc.fit(x_train,y_train)
rc_pred = rc.predict(x_test)
rc_acc = accuracy_score (y_test,rc_pred) * 100
rc_acc


# In[26]:


rc_test1 = rc.predict (x_train)
rc_acc1 = accuracy_score (y_train , rc_test1) * 100
rc_acc1


# ###  Randomn Forest
# Training Score = 91
# Test Score = 81

# ### Support vector machine

# In[27]:


from sklearn.svm import SVC
sv = SVC(kernel='poly')
sv = sv.fit(x_train , y_train)
sv_pred = sv.predict(x_test)
sv_acc = round(accuracy_score (y_test , sv_pred)*100)
sv_acc


# In[28]:


sv_pred1 = sv.predict(x_train)
sv_acc1 = round(accuracy_score (y_train , sv_pred1) * 100)
sv_acc1


# ###  SVC
# Training Score = 86
# Test Score = 85

# ### Logistic Regression and SVC has highest accuracy

# In[29]:


P_Survived = sv.predict(train_x)
P_acc = round(accuracy_score (train_y,P_Survived)*100)
P_acc


# ### P_Survived = Predicted Survived

# In[30]:


P_Survived = pd.DataFrame(P_Survived,columns = ['P_Survived'])
df_new = pd.concat([df,y],axis=1)
df_new = pd.concat ([df_new,P_Survived],axis=1)
df_new

