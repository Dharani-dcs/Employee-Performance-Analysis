#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# In[2]:


data=pd.read_csv("employee analysis.csv")


# ##### PREPROCESSING THE DATASET
# 

# In[3]:


data.head()  #print the first five rows 


# In[4]:


data.describe()  #to check the scatterness of the column data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


data.select_dtypes('object').nunique()#number of unique items present in the categorical column.


# In[6]:


data.info() #allow to learn the shape of the object type of the data


# In[7]:


data.describe(include=['O']) # To see the Distribution of Categorical features


# In[8]:


data.isna().values.any() # To find out NaN values


# In[9]:


data.isnull().values.any() # To find out Null values


# In[11]:


data.corr().T


# ##### Analysis of department wise performance

# In[13]:


# A new pandas Dataframe is created to analyze department wise performance.
dept = data.iloc[:,[5,27]].copy()
dept_per = dept.copy()
dept_per


# In[14]:


# Finding out the mean performance of all the departments and plotting its bar graph using seaborn.
dept_per.groupby(by='EmpDepartment')['PerformanceRating'].mean()


# In[15]:


plt.figure(figsize=(10,4.5))
sns.barplot(dept_per['EmpDepartment'],dept_per['PerformanceRating'])


# In[16]:


# Analyze each department separately
dept_per.groupby(by='EmpDepartment')['PerformanceRating'].value_counts()


# In[17]:


# Creating a new dataframe to analyze each department separately
department = pd.get_dummies(dept_per['EmpDepartment'])
performance = pd.DataFrame(dept_per['PerformanceRating'])
dept_rating = pd.concat([department,performance],axis=1)


# In[18]:


# Plotting a separate bar graph for performance of each department using seaborn
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.barplot(dept_rating['PerformanceRating'],dept_rating['Sales'])
plt.subplot(2,3,2)
sns.barplot(dept_rating['PerformanceRating'],dept_rating['Development'])
plt.subplot(2,3,3)
sns.barplot(dept_rating['PerformanceRating'],dept_rating['Research & Development'])
plt.subplot(2,3,4)
sns.barplot(dept_rating['PerformanceRating'],dept_rating['Human Resources'])
plt.subplot(2,3,5)
sns.barplot(dept_rating['PerformanceRating'],dept_rating['Finance'])
plt.subplot(2,3,6)
sns.barplot(dept_rating['PerformanceRating'],dept_rating['Data Science'])
plt.show()


# ##### feature selection

# In[20]:


sns.heatmap(data.corr())


# In[21]:


data=data.drop('EmpNumber',axis=1)
data.head()


# In[44]:


data.columns


# ##### encoding the predictors

# In[22]:


data1=pd.get_dummies(data,drop_first=True)
data1.head()


# In[23]:


data1.shape


# In[24]:


data1.columns


# In[25]:


# Here we have selected only the important columns
y = data1.PerformanceRating
#X = data.iloc[:,0:-1]  All predictors were selected it resulted in dropping of accuracy.
X = data1.iloc[:,[4,5,9,16,20,21,22,23,24]] # Taking only variables with correlation coeffecient greater than 0.1
X.head()


# In[26]:


# Splitting into train and test for calculating the accuracy
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# In[27]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[28]:


X_train.shape


# In[29]:


X_test.shape


# ##### logistic regression

# In[30]:


# Training the model
from sklearn.linear_model import LogisticRegression
model_logr = LogisticRegression()
model_logr.fit(X_train,y_train)


# In[31]:



# Predicting the model
y_predict_log = model_logr.predict(X_test)


# In[32]:


# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_log))
print(classification_report(y_test,y_predict_log))


# In[33]:


confusion_matrix(y_test,y_predict_log)


# ##### K-Nearest Neighbor

# In[34]:


# Training the model
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean') # Maximum accuracy for n=10
model_knn.fit(X_train,y_train)


# In[35]:


# Predicting the model
y_predict_knn = model_knn.predict(X_test)


# In[36]:


# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_knn))
print(classification_report(y_test,y_predict_knn))


# In[37]:


confusion_matrix(y_test,y_predict_knn)


# ##### RANDOMFOREST CLASSIFIER

# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
model = RandomForestClassifier(random_state=10)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(classification_report(y_test,y_predict))
print("Accuracy Score : ",accuracy_score(y_test,y_predict)*100,"%")
print(" ")
print(" ")


# In[39]:


confusion_matrix(y_test,y_predict)


# ##### Naive Bayes Bernoulli

# In[40]:


# Training the model
from sklearn.naive_bayes import BernoulliNB
model_nb = BernoulliNB()
model_nb.fit(X_train,y_train)


# In[41]:


# Predicting the model
y_predict_nb = model_nb.predict(X_test)


# In[42]:


# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_nb))
print(classification_report(y_test,y_predict_nb))


# In[43]:


confusion_matrix(y_test,y_predict_nb)


# ##### support vector machine

# In[44]:


# Training the model
from sklearn.svm import SVC
rbf_svc = SVC(kernel='rbf', C=100, random_state=10).fit(X_train,y_train)


# In[45]:


# Predicting the model
y_predict_svm = rbf_svc.predict(X_test)


# In[46]:


# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_svm))
print(classification_report(y_test,y_predict_svm))
confusion_matrix(y_test,y_predict_svm)

