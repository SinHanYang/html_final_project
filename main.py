#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import pandas as pd
import math
import scipy


# # Collect data

# ### demographics

# In[117]:


#user= dempgraphics?
users=pd.read_csv('demographics.csv')
users.columns=['Customer ID','Count','Gender','Age','Under_30','Senior_Citizen','Married','Dependents','Number_of_Dependents']
# do we still need Under 30/ Senior Citizen, or we need to group the age ?
users.drop(['Count'],axis=1,inplace=True)
users.drop(['Under_30','Senior_Citizen','Dependents'],axis=1,inplace=True)
users


# ### location

# In[118]:


location=pd.read_csv('location.csv')
location.columns=['Customer ID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longtitude']
# Country, State are the same, latitude/longtitude's information is in Zip code
location.drop(['Count','Country','State','Lat Long','Latitude','Longtitude'],axis=1,inplace=True)
location.drop(['City'],axis=1,inplace=True)


# ### population

# In[119]:


population=pd.read_csv('population.csv')
population.columns=['ID','Zip Code','Population']
population.drop(['ID'],axis=1,inplace=True)
population


# ### merge location and population ,and join users

# In[120]:


location=pd.merge(location,population,on='Zip Code')
users=pd.merge(users,location,on='Customer ID',how='outer')
users


# satisfaction

# In[121]:


satisfaction=pd.read_csv('satisfaction.csv')
satisfaction.columns=['Customer ID','score']
users=pd.merge(users,satisfaction,on='Customer ID',how='outer')


# services

# In[122]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
services=pd.read_csv('services.csv')
services.columns=['Customer ID','Count','Quarter','Referred_a_Friend','Number of Referrals','Tenure in Months','Offer','Phone_Service','Avg Monthly Long Distance Charges','Multiple_Lines','Internet_Service','Internet_Type','Avg Monthly GB Download','Online_Security','Online_Backup','Device_Protection_Plan','Premium_Tech_Support','Streaming_TV','Streaming_Movies','Streaming_Music','Unlimited_Data','Contract','Paperless_Billing','Payment_Method','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']
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
#users = pd.concat((users,pd.get_dummies(users.Referred_a_friend,prefix='Referred_a_friend')),1)
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

users.drop(['Married','Gender','Phone_Service','Multiple_Lines','Internet_Service','Offer','Internet_Type','Online_Security','Online_Backup','Device_Protection_Plan','Premium_Tech_Support','Streaming_TV','Streaming_Movies','Streaming_Music','Unlimited_Data','Contract','Paperless_Billing','Payment_Method'],axis=1,inplace=True)
'''
df["Married"]=pd.util.hash_array(df["Married"].to_numpy())
df["Gender"]=pd.util.hash_array(df["Gender"].to_numpy())
df["Under 30"]=pd.util.hash_array(df["Under 30"].to_numpy())
df["Senior Citizen"]=pd.util.hash_array(df["Senior Citizen"].to_numpy())
df["Dependents"]=pd.util.hash_array(df["Dependents"].to_numpy())
df["City"]=pd.util.hash_array(df["City"].to_numpy())
#df["Referred a friend"]=pd.util.hash_array(df["Referred a friend"].to_numpy())
df["Offer"]=pd.util.hash_array(df["Offer"].to_numpy())
df["Phone Service"]=pd.util.hash_array(df["Phone Service"].to_numpy())
df["Multiple Lines"]=pd.util.hash_array(df["Multiple Lines"].to_numpy())
df["Internet Service"]=pd.util.hash_array(df["Internet Service"].to_numpy())
df["Internet Type"]=pd.util.hash_array(df["Internet Type"].to_numpy())
df["Online Security"]=pd.util.hash_array(df["Online Security"].to_numpy())
df["Online Backup"]=pd.util.hash_array(df["Online Backup"].to_numpy())
df["Device Protection Plan"]=pd.util.hash_array(df["Device Protection Plan"].to_numpy())
df["Premium Tech Support"]=pd.util.hash_array(df["Premium Tech Support"].to_numpy())
df["Streaming TV"]=pd.util.hash_array(df["Streaming TV"].to_numpy())
df["Streaming Movies"]=pd.util.hash_array(df["Streaming Movies"].to_numpy())
df["Streaming Music"]=pd.util.hash_array(df["Streaming Music"].to_numpy())
df["Unlimited Data"]=pd.util.hash_array(df["Unlimited Data"].to_numpy())
df["Contract"]=pd.util.hash_array(df["Contract"].to_numpy())
df["Paperless Billing"]=pd.util.hash_array(df["Paperless Billing"].to_numpy())
df["Payment Method"]=pd.util.hash_array(df["Payment Method"].to_numpy())
'''
users


# status

# In[123]:


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

# In[124]:


from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from libsvm.svmutil import *


# In[125]:


features=list(train)
#print(features)
features.remove('Customer ID')
features.remove('Churn Category')
features.remove('Referred_a_Friend')
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


# In[126]:


# 10% vaildation 
x_test,x_train,y_test,y_train = train_test_split(train_imputed,train.loc[:, 'Churn Category'],test_size=0.90, random_state=0)


# ### SVM_linear 

# In[127]:


for g in range(1):
    for c in range(1,2):
        svm_model_linear = SVC(kernel = 'rbf',gamma=math.pow(10,g), C = math.pow(10,c)).fit(x_train, y_train)
        svm_predictions = svm_model_linear.predict(x_test)
        accuracy = svm_model_linear.score(x_test, y_test)
        print("sklearn_svm",g,c,accuracy)


# In[128]:


svm_model_linear = SVC(kernel = 'rbf',gamma=math.pow(10,0), C = math.pow(10,1)).fit(train_imputed, train.loc[:,'Churn Category'])


# In[129]:



y_train_num=y_train.to_numpy()
y_train_num=y_train_num.astype(np.int)
x_train_num=x_train.to_numpy()
prob=svm_problem(y_train_num,x_train_num)
param = svm_parameter('-t 2 -c 10')
libsvm_train=svm_train(prob,param)
y_test_num=y_test.to_numpy()
y_test_num=y_test_num.astype(np.int)
x_test_num=x_test.to_numpy()
svm_predict(y_train_num,x_train_num,libsvm_train)


# In[148]:


y_test_num=y_test_num.astype(np.int)
p_label, p_acc, p_val=svm_predict(y_test_num,x_test_num,libsvm_train)

y_train_all=train.loc[:,'Churn Category'].to_numpy()
y_train_all=y_train_all.astype(np.int)
x_train_all=train_imputed.to_numpy()
prob_all=svm_problem(y_train_all,x_train_all)
libsvm_train=svm_train(prob_all,param)


# ### decision tree

# In[131]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 1).fit(x_train, y_train)
dtree_predictions = dtree_model.predict(x_test)
accuracy = dtree_model.score(x_test, y_test)
print("dtree",accuracy)
dtree_model = DecisionTreeClassifier(max_depth = 1).fit(train_imputed,train.loc[:,'Churn Category'])


# ### knn 

# In[132]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10).fit(x_train, y_train)
accuracy = knn.score(x_test, y_test)
print("knn",accuracy)


# ### RandomForest + ada boost

# In[144]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

rf = RandomForestClassifier(n_estimators = 1500, random_state = 0,n_jobs=-1)



rf.fit(x_train, y_train)
print("rf")
rf.score(x_test, y_test)
rf.fit(train_imputed,train.loc[:, 'Churn Category'])


# In[134]:


ada = AdaBoostClassifier(n_estimators = 1000)
ada.fit(x_train, y_train)
print("ada")
ada.score(x_test, y_test)
ada.fit(train_imputed,train.loc[:, 'Churn Category'])


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

# In[184]:


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

dftest['Churn Category']=ada.predict(test_imputed)


# ### Output result

# In[185]:


dftest.columns=['Customer ID','Churn Category']
submiss=pd.DataFrame(dftest)
submiss.to_csv('submission.csv',index=False)

