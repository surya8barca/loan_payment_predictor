#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('loan_data.csv')


# In[4]:


df.head()


# In[5]:


fig=sns.pairplot(df)


# In[9]:


fig.savefig('pairplots2.jpg')


# In[12]:


df['purpose'].unique()


# In[15]:


df


# In[21]:


fig=df[df['credit.policy']==1]['fico'].hist(label='Credit_policy=1',alpha=0.6)
fig=df[df['credit.policy']==0]['fico'].hist(label='Credit_policy=0',alpha=0.6)
plt.legend()


# In[22]:


fig=df[df['not.fully.paid']==1]['fico'].hist(label='Not Fully Paid=1',alpha=0.6)
fig=df[df['not.fully.paid']==0]['fico'].hist(label='Not Fully Paid=0',alpha=0.6)
plt.legend()


# In[28]:


plt.figure(figsize=(11,8))
fig=sns.countplot(x='purpose',data=df,hue='not.fully.paid')


# In[29]:


sns.jointplot(x='fico',y='int.rate',data=df)


# In[32]:


#creating dummies for catagorical data i.e purpose


# In[33]:


cf=['purpose']


# In[34]:


final_data=pd.get_dummies(df,columns=cf,drop_first=True)


# In[35]:


final_data.head()


# In[36]:


#train test split the data


# In[39]:


x=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)


# In[42]:


#set the model using decision tree


# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


dtc=DecisionTreeClassifier()


# In[45]:


dtc.fit(x_train,y_train)


# In[46]:


pred1=dtc.predict(x_test)


# In[47]:


pred1


# In[48]:


#evaluate data


# In[49]:


from sklearn.metrics import confusion_matrix,classification_report


# In[53]:


print(confusion_matrix(y_test,pred1))
print()
print(classification_report(y_test,pred1))


# In[55]:


sns.lineplot(y_test,pred1)


# In[52]:


#now using random forest with n_estimator as 500


# In[56]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


rfc=RandomForestClassifier(n_estimators=500)


# In[64]:


rfc.fit(x_train,y_train)


# In[65]:


pred2=rfc.predict(x_test)


# In[66]:


pred2


# In[61]:


#evaluate data


# In[67]:


print(confusion_matrix(y_test,pred2))
print()
print(classification_report(y_test,pred2))


# In[68]:


sns.lineplot(y_test,pred2)


# In[69]:


#comparison


# In[74]:


sns.lineplot(pred1,pred2)


# In[ ]:


#the end

