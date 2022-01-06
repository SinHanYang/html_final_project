#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import math
import scipy


# # Collect data

# ### demographics

# In[11]:


#user= dempgraphics?
users=pd.read_csv('demographics.csv')
users.columns=['Customer ID','Count','Gender','Age','Under_30','Senior_Citizen','Married','Dependents','Number_of_Dependents']
# do we still need Under 30/ Senior Citizen, or we need to group the age ?
users.drop(['Count'],axis=1,inplace=True)
users.drop(['Under_30','Senior_Citizen','Dependents'],axis=1,inplace=True)
users


# ### location

# In[12]:


location=pd.read_csv('location.csv')
location.columns=['Customer ID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longtitude']
# Country, State are the same, latitude/longtitude's information is in Zip code
location.drop(['Count','Country','State','Lat Long','Latitude','Longtitude'],axis=1,inplace=True)
location.drop(['City'],axis=1,inplace=True)


# ### population

# In[13]:


population=pd.read_csv('population.csv')
population.columns=['ID','Zip Code','Population']
population.drop(['ID'],axis=1,inplace=True)
population


# ### merge location and population ,and join users

# In[14]:


location=pd.merge(location,population,on='Zip Code')
users=pd.merge(users,location,on='Customer ID',how='outer')
users


# satisfaction

# In[15]:


satisfaction=pd.read_csv('satisfaction.csv')
satisfaction.columns=['Customer ID','score']
users=pd.merge(users,satisfaction,on='Customer ID',how='outer')


# services

# In[16]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
services=pd.read_csv('services.csv')
services.columns=['Customer ID','Count','Quarter','Referred_a_friend','Number of Referrals','Tenure in Months','Offer','Phone_Service','Avg Monthly Long Distance Charges','Multiple_Lines','Internet_Service','Internet_Type','Avg Monthly GB Download','Online_Security','Online_Backup','Device_Protection_Plan','Premium_Tech_Support','Streaming_TV','Streaming_Movies','Streaming_Music','Unlimited_Data','Contract','Paperless_Billing','Payment_Method','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']
# Quarter are always Q3
services.drop(['Count','Quarter'],axis=1,inplace=True)
users=pd.merge(users,services,on='Customer ID',how='outer')
#columns_to_encode = [1,3,4,5,6,8,12,13,14,15,16,17,18,20,21,22,23,34,35,36,27,28,29]
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')

# give null a category?
#users.loc[users.Married.isnull(),'Married']='None'
users = pd.concat((users,pd.get_dummies(users.Married,prefix='Married')),1)
users = pd.concat((users,pd.get_dummies(users.Gender,prefix='Gender')),1)
#users = pd.concat((users,pd.get_dummies(users.Under_30,prefix='Under_30')),1)
#users = pd.concat((users,pd.get_dummies(users.Senior_Citizen,prefix='Senior_Citizen')),1)
#users = pd.concat((users,pd.get_dummies(users.Dependents,prefix='Dependents')),1)
#users = pd.concat((users,pd.get_dummies(users.City,prefix='City')),1)
users = pd.concat((users,pd.get_dummies(users.Referred_a_friend,prefix='Referred_a_friend')),1)
users = pd.concat((users,pd.get_dummies(users.Offer,prefix='Offer')),1)
users = pd.concat((users,pd.get_dummies(users.Phone_Service,prefix='Phone_Service')),1)
users = pd.concat((users,pd.get_dummies(users.Multiple_Lines,prefix='Multiple_Lines')),1)
users = pd.concat((users,pd.get_dummies(users.Internet_Service,prefix='Internet_Service')),1)
users = pd.concat((users,pd.get_dummies(users.Internet_Type,prefix='Internet_Type')),1)
users = pd.concat((users,pd.get_dummies(users.Online_Security,prefix='Online_Security')),1)
users = pd.concat((users,pd.get_dummies(users.Online_Backup,prefix='Online_Backup')),1)
users = pd.concat((users,pd.get_dummies(users.Device_Protection_Plan,prefix='Device_Protection_Plan')),1)
users = pd.concat((users,pd.get_dummies(users.Premium_Tech_Support,prefix='Premium_Tech_Support')),1)
users = pd.concat((users,pd.get_dummies(users.Streaming_TV,prefix='Streaming_TV')),1)
users = pd.concat((users,pd.get_dummies(users.Streaming_Movies,prefix='Streaming_Movies')),1)
users = pd.concat((users,pd.get_dummies(users.Streaming_Music,prefix='Streaming_Music')),1)
users = pd.concat((users,pd.get_dummies(users.Unlimited_Data,prefix='Unlimited_Data')),1)
users = pd.concat((users,pd.get_dummies(users.Contract,prefix='Contract')),1)
users = pd.concat((users,pd.get_dummies(users.Paperless_Billing,prefix='Paperless_Billing')),1)
users = pd.concat((users,pd.get_dummies(users.Payment_Method,prefix='Payment_Method')),1)

users.drop(['Married','Gender','Phone_Service','Multiple_Lines','Internet_Service','Referred_a_friend','Offer','Internet_Type','Online_Security','Online_Backup','Device_Protection_Plan','Premium_Tech_Support','Streaming_TV','Streaming_Movies','Streaming_Music','Unlimited_Data','Contract','Paperless_Billing','Payment_Method'],axis=1,inplace=True)

users


# status

# In[17]:


status=pd.read_csv('status.csv')
status.columns=['Customer ID','Churn Category']
status['ans'] = status['Churn Category']
status.loc[status.ans=='No Churn','ans']='0'
status.loc[status.ans=='Competitor','ans']='1'
status.loc[status.ans=='Dissatisfaction','ans']='2'
status.loc[status.ans=='Attitude','ans']='3'
status.loc[status.ans=='Price','ans']='4'
status.loc[status.ans=='Other','ans']='5'
status.drop(['Churn Category'],axis=1,inplace=True)
status.rename(columns={'ans':'Churn Category'}, inplace=True)
#final train data
train=pd.merge(status,users,on='Customer ID',how='left')


# # Train Model

# In[18]:


from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from libsvm.svmutil import *


# In[19]:


features=list(train)
#print(features)
features.remove('Customer ID')
features.remove('Churn Category')
#features.remove('Referred_a_Friend')
#features.remove('Offer')
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(train.loc[:, features])
train_imputed = imputer.fit_transform(train.loc[:, features])
df=pd.DataFrame(train_imputed)
poly = PolynomialFeatures(degree=2, interaction_only=False)
train_imputed = pd.DataFrame(poly.fit_transform(df))
scalar=StandardScaler()
train_imputed=scalar.fit_transform(train_imputed)
train_imputed = pd.DataFrame(train_imputed)
train_imputed


# In[20]:


# 10% vaildation 
x_test,x_train,y_test,y_train = train_test_split(train_imputed,train.loc[:, 'Churn Category'],test_size=0.90, random_state=0)


# In[21]:


ros = RandomOverSampler(random_state=0)
#x_train, y_train = ros.fit_resample(x_train, y_train)
x_train, y_train = SMOTE().fit_resample(x_train, y_train)
#x_train, y_train = SVMSMOTE().fit_resample(x_train, y_train)

# ### SVM_linear 

# In[23]:



y_train_num=y_train.to_numpy()
y_train_num=y_train_num.astype(np.int)
x_train_num=x_train.to_numpy()
prob=svm_problem(y_train_num,x_train_num)
C=0.1
for i in range(1):
    print("C:",C)
    param = svm_parameter(f'-t 0 -c {C} -q')
    libsvm_train=svm_train(prob,param)
    y_test_num=y_test.to_numpy()
    y_test_num=y_test_num.astype(np.int)
    x_test_num=x_test.to_numpy()
    p_label, p_acc, p_val=svm_predict(y_train_num,x_train_num,libsvm_train)




    y_test_num=y_test_num.astype(np.int)
    p_label, p_acc, p_val=svm_predict(y_test_num,x_test_num,libsvm_train)
    score=f1_score(y_test_num,p_label,average='macro')
    print(score)
    matrix=confusion_matrix(y_test_num,p_label)
    print(matrix)
    C*=3.16
'''
x_all, y_all = SMOTE().fit_resample(train_imputed, train.loc[:,'Churn Category'])
y_train_all=y_all.to_numpy()
y_train_all=y_train_all.astype(np.int)
x_train_all=x_all.to_numpy()
prob_all=svm_problem(y_train_all,x_train_all)
libsvm_train=svm_train(prob_all,param)
'''
'''
# ### decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 1).fit(x_train, y_train)
dtree_predictions = dtree_model.predict(x_test)
matrix=confusion_matrix(y_test_num,dtree_predictions)
print(matrix)
accuracy = dtree_model.score(x_test, y_test)
print("dtree",accuracy)
dtree_model = DecisionTreeClassifier(max_depth = 1).fit(train_imputed,train.loc[:,'Churn Category'])


# ### knn 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10).fit(x_train, y_train)
accuracy = knn.score(x_test, y_test)
print("knn",accuracy)

'''
# ### RandomForest + ada boost

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
for tree in range(10,11):
    for d in range(8,11):
        rf = RandomForestClassifier(n_estimators = tree*1000, max_depth=d,oob_score=True)
        rf.fit(x_train,y_train)
        print(f"tree:{tree*1000},d:{d},acc:{rf.oob_score_}")
        dtree_predictions = rf.predict(x_test)
        matrix=confusion_matrix(y_test,dtree_predictions)
        print(matrix)
        score=f1_score(y_test,dtree_predictions,average='macro')
        print(score)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier 
est=240
for i in range(1):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators =est,learning_rate=0.2)
    ada.fit(x_train, y_train)
    print("ada,est:",est,ada.score(x_test,y_test))
    pre=ada.predict(x_test)
    score=f1_score(y_test,pre,average='macro')
    print("f1 score",score)
    matrix=confusion_matrix(y_test,pre)
    print(matrix)
    est+=20

# ### Prediction

# ### handle test

# In[ ]:


testID=pd.read_csv('Test_IDs.csv')
testID.columns=['Customer ID']
dftest=pd.DataFrame(testID)
test=pd.merge(dftest,users,on='Customer ID',how='left')
#print(test)
test_imputed = imputer.fit_transform(test.loc[:, features])
df=pd.DataFrame(test_imputed)
poly = PolynomialFeatures(degree=2, interaction_only=False)
test_imputed = pd.DataFrame(poly.fit_transform(df))
scalar=StandardScaler()
test_imputed=scalar.fit_transform(test_imputed)
test_imputed = pd.DataFrame(test_imputed)
#print(train_imputed.shape)


# ### SVM prediction

# In[ ]:


#dftest['Churn Category']=svm_model_linear.predict(test_imputed)
total_rows=test_imputed.shape[0]
test_imputed=test_imputed.to_numpy()
fake_y=np.zeros(total_rows)
p_label, p_acc, p_val=svm_predict(fake_y,test_imputed,libsvm_train)
p_label = list(map(int, p_label))
dftest['Churn Category']=p_label


# ### not svm

# In[ ]:


#dftest['Churn Category']=dtree_model.predict(test_imputed)

#dftest['Churn Category']=knn.predict(test_imputed)

#dftest['Churn Category']=rf.predict(test_imputed)

#dftest['Churn Category']=ada.predict(test_imputed)


# ### Output result

# In[ ]:


dftest.columns=['Customer ID','Churn Category']
submiss=pd.DataFrame(dftest)
submiss.to_csv('submission.csv',index=False)

