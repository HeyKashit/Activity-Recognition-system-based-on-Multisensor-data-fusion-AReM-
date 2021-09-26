#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In this problem, we classify the activities of humans based on time series obtained by a Wireless Sensor Network. The dataset contains 7 folders that represent seven types of activities. In each folder, there are multiple files each of which represents an instant of a human performing an activity. Each file containis 6 time series collected from activities of the same person, which are called avg rss12, var rss12, avg rss13, var rss13, vg rss23, and ar rss23. There are 88 instances in the dataset, each of which contains 6 time series and each time series has 480 consecutive values

# In[2]:


# for Bending1.
df1=pd.read_csv('bending1\dataset1.csv',skiprows=4)
df2=pd.read_csv('bending1\dataset2.csv',skiprows=4)
df3=pd.read_csv('bending1\dataset3.csv',skiprows=4)
df4=pd.read_csv('bending1\dataset4.csv',skiprows=4)
df5=pd.read_csv('bending1\dataset5.csv',skiprows=4)
df6=pd.read_csv('bending1\dataset6.csv',skiprows=4)

# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6],axis=0)

# now adding bending columns in the last as it is our y label.
df_bending1 = df_ben.assign(value ="bending1")

# now checking reuired dataframe
df_bending1


# In[3]:


# for bending2.

df1=pd.read_csv('bending2\dataset1.csv',skiprows=4)
df2=pd.read_csv('bending2\dataset2.csv',skiprows=4)
df3=pd.read_csv('bending2\dataset3.csv',skiprows=4)
df4=pd.read_csv('bending2\dataset4.csv',skiprows=4)
df5=pd.read_csv('bending2\dataset5.csv',skiprows=4)
df6=pd.read_csv('bending2\dataset6.csv',skiprows=4)

# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6],axis=0)

# now adding bending columns in the last as it is our y label.
df_bending2 = df_ben.assign(value ="bending2")

# now checking reuired dataframe
df_bending2


# In[4]:


# for cycling

df1=pd.read_csv('cycling\dataset1.csv',skiprows=4)
df2=pd.read_csv('cycling\dataset2.csv',skiprows=4)
df3=pd.read_csv('cycling\dataset3.csv',skiprows=4)
df4=pd.read_csv('cycling\dataset4.csv',skiprows=4)
df5=pd.read_csv('cycling\dataset5.csv',skiprows=4)
df6=pd.read_csv('cycling\dataset6.csv',skiprows=4)
df7=pd.read_csv('cycling\dataset7.csv',skiprows=4)

# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6,df7],axis=0)

# now adding bending columns in the last as it is our y label.
df_cycling = df_ben.assign(value ="cycling")

# now checking reuired dataframe
df_cycling


# In[5]:


# for lying

df1=pd.read_csv('lying\dataset1.csv',skiprows=4)
df2=pd.read_csv('lying\dataset11.csv',skiprows=4)
df3=pd.read_csv('lying\dataset8.csv',skiprows=4)
df4=pd.read_csv('lying\dataset4.csv',skiprows=4)
df5=pd.read_csv('lying\dataset13.csv',skiprows=4)
df6=pd.read_csv('lying\dataset6.csv',skiprows=4)


# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6],axis=0)

# now adding bending columns in the last as it is our y label.
df_lying = df_ben.assign(value ="lying")

# now checking reuired dataframe
df_lying


# In[6]:


# for sitting

df1=pd.read_csv('sitting\dataset1.csv',skiprows=4)
df2=pd.read_csv('sitting\dataset2.csv',skiprows=4)
df3=pd.read_csv('sitting\dataset3.csv',skiprows=4)
df4=pd.read_csv('sitting\dataset4.csv',skiprows=4)
df5=pd.read_csv('sitting\dataset5.csv',skiprows=4)
df6=pd.read_csv('sitting\dataset6.csv',skiprows=4)
df7=pd.read_csv('sitting\dataset7.csv',skiprows=4)

# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6,df7],axis=0)

# now adding bending columns in the last as it is our y label.
df_sitting = df_ben.assign(value ="sitting")

# now checking reuired dataframe
df_sitting


# In[7]:


# for standing

df1=pd.read_csv('standing\dataset1.csv',skiprows=4)
df2=pd.read_csv('standing\dataset2.csv',skiprows=4)
df3=pd.read_csv('standing\dataset3.csv',skiprows=4)
df4=pd.read_csv('standing\dataset4.csv',skiprows=4)
df5=pd.read_csv('standing\dataset5.csv',skiprows=4)
df6=pd.read_csv('standing\dataset6.csv',skiprows=4)
df7=pd.read_csv('standing\dataset7.csv',skiprows=4)

# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6,df7],axis=0)

# now adding bending columns in the last as it is our y label.
df_standing = df_ben.assign(value ="standing")

# now checking reuired dataframe
df_standing


# In[8]:


# for walking

df1=pd.read_csv('walking\dataset1.csv',skiprows=4)
df2=pd.read_csv('walking\dataset2.csv',skiprows=4)
df3=pd.read_csv('walking\dataset3.csv',skiprows=4)
df4=pd.read_csv('walking\dataset4.csv',skiprows=4)
df5=pd.read_csv('walking\dataset5.csv',skiprows=4)
df6=pd.read_csv('walking\dataset6.csv',skiprows=4)
df7=pd.read_csv('walking\dataset7.csv',skiprows=4)

# now using concat function to add them and make a single dataFrame.
df_ben=pd.concat([df1,df2,df3,df4,df5,df6,df7],axis=0)

# now adding bending columns in the last as it is our y label.
df_walking = df_ben.assign(value ="walking")

# now checking reuired dataframe
df_walking


# In[9]:


# now making final data to work with
final_data=pd.concat([df_bending1,df_bending2,df_cycling,df_lying,df_sitting,df_standing,df_walking])
final_data


# In[10]:


df = final_data.sample(frac =1)
df


# In[11]:


# now checking n unique values in the data.
df.value.unique()


# In[12]:


df.isnull().sum()


# In[13]:


# dropping the null values
df1=df.dropna()
df1.shape


# In[14]:


df1.columns


# In[15]:


df1.isnull().sum()


# In[16]:


df1.describe()
df.dtypes
df1['value'] = df1['value'].map({'bending1': 1, 'bending2': 0,'lying':2,'walking':3 , 'standing': 4,'sitting':5,'cycling':6})
df1.to_csv('newdata_name.csv')


# In[ ]:





# In[17]:


df2=df1.drop(['# Columns: time'],axis=1)
# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df2:
    if plotnumber<=6 :
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(df2[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# In[18]:


df1.drop(['# Columns: time'],axis=1,inplace=True)

# checking for the outliers


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df2, width= 0.5,ax=ax,  fliersize=3)


# we have a lot of outliers in the data . so, we need to remove it

# In[910]:



q = df1['var_rss12'].quantile(0.60)
df1 = df1[df1['var_rss12']<q]

q = df1['var_rss12'].quantile(0.80)
df1 = df1[df1['var_rss12']<q]

q = df1['var_rss23'].quantile(0.70)
df1 = df1[df1['var_rss23']<q]

q = df1['avg_rss12'].quantile(0.90)
df1 = df1[df1['avg_rss12']<q]

q = df1['var_rss13'].quantile(0.90)
df1 = df1[df1['var_rss13']<q]


# In[911]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df1, width= 0.5,ax=ax,  fliersize=3)


# In[912]:


x=df1.drop(['value'],axis=1)
y=df1['value']
x.columns


# In[913]:


x.isnull().sum()


# In[914]:


y


# In[915]:


df1.shape


# In[916]:


from sklearn.preprocessing import StandardScaler 
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)


# In[917]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = x.columns

#let's check the values
vif


# All the VIF values are less than 5 and are very low. That means no multicollinearity. Now, we can go ahead with fitting our data to the model. Before that, let's split our data in test and training set.

# # Splitting data into Training and Testing Set

# In[918]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# # 1 model using logistic regression

# In[919]:


# making model 
from sklearn.linear_model  import  LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)


# In[920]:


y_pred = log_reg.predict(X_test)


# In[921]:


from sklearn.metrics import accuracy_score,classification_report
report = classification_report(y_test, y_pred)
print(report)
print("Accuracy of the RandomForest Model is:",accuracy_score(y_test,y_pred)*100,"%")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # using 2 model

# In[922]:


# using 2 model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
model=RandomForestClassifier()
model.fit(X_train,y_train)
# accuracy of RandomForest Model
y_predxgb = model.predict(X_test)
report = classification_report(y_test, y_predxgb)
print(report)
print("Accuracy of the RandomForest Model is:",accuracy_score(y_test,y_predxgb)*100,"%")


# In[923]:


import pickle
filename='Finalmodel.sav'
pickle.dump(model,open(filename,'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




