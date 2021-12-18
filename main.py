#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd


# # Collect data

# ### demographics

# In[35]:


#user= dempgraphics?
users=pd.read_csv('demographics.csv')
users.columns=['Customer ID','Count','Gender','Age','Under_30','Senior_Citizen','Married','Dependents','Number_of_Dependents']
# do we still need Under 30/ Senior Citizen, or we need to group the age ?
users.drop(['Count'],axis=1,inplace=True)
users.drop(['Under_30','Senior_Citizen','Dependents'],axis=1,inplace=True)
users


# ### location

# In[36]:


location=pd.read_csv('location.csv')
location.columns=['Customer ID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longtitude']
# Country, State are the same, latitude/longtitude's information is in Zip code
location.drop(['Count','Country','State','Lat Long','Latitude','Longtitude'],axis=1,inplace=True)
location.drop(['City'],axis=1,inplace=True)


# ### population

# In[37]:


population=pd.read_csv('population.csv')
population.columns=['ID','Zip Code','Population']
population.drop(['ID'],axis=1,inplace=True)
population


# ### merge location and population ,and join users

# In[38]:


location=pd.merge(location,population,on='Zip Code')
users=pd.merge(users,location,on='Customer ID',how='outer')
users


# satisfaction

# In[39]:


satisfaction=pd.read_csv('satisfaction.csv')
satisfaction.columns=['Customer ID','score']
users=pd.merge(users,satisfaction,on='Customer ID',how='outer')


# services

# In[40]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
services=pd.read_csv('services.csv')
services.columns=['Customer ID','Count','Quarter','Referred_a_Friend','Number of Referrals','Tenure in Months','Offer','Phone_Service','Avg Monthly Long Distance Charges','Multiple_Lines','Internet_Service','Internet_Type','Avg Monthly GB Download','Online_Security','Online_Backup','Device_Protection_Plan','Premium_Tech_Support','Streaming_TV','Streaming_Movies','Streaming_Music','Unlimited_Data','Contract','Paperless_Billing','Payment_Method','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']
# Quarter are always Q3
services.drop(['Count','Quarter'],axis=1,inplace=True)
users=pd.merge(users,services,on='Customer ID',how='outer')
#columns_to_encode = [1,3,4,5,6,8,12,13,14,15,16,17,18,20,21,22,23,34,35,36,27,28,29]
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
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

# In[41]:


status=pd.read_csv('status.csv')
status.columns=['Customer ID','Churn Category']
#final train data
train=pd.merge(status,users,on='Customer ID',how='left')


# # Train Model

# In[42]:


from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC 


# In[43]:


features=list(train)
#print(features)
features.remove('Customer ID')
features.remove('Churn Category')
features.remove('Referred_a_Friend')
#features.remove('Offer')
'''
features.remove('Under 30')
features.remove('Married')
features.remove('Senior Citizen')
features.remove('Dependents')
features.remove('City')
features.remove('Referred a Friend')
features.remove('Offer')
features.remove('Phone Service')
features.remove('Multiple Lines')
features.remove('Internet Service')
features.remove('Internet Type')
features.remove('Online Security')
features.remove('Online Backup')
features.remove('Device Protection Plan')
features.remove('Premium Tech Support')
features.remove('Streaming TV')
features.remove('Streaming Movies')
features.remove('Streaming Music')
features.remove('Unlimited Data')
features.remove('Contract')
features.remove('Paperless Billing')
features.remove('Payment Method')
'''
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(train.loc[:, features])
train_imputed = imputer.fit_transform(train.loc[:, features])
df=pd.DataFrame(train_imputed)
poly = PolynomialFeatures(degree=2, interaction_only=True)
train_imputed = pd.DataFrame(poly.fit_transform(df))
print(train_imputed.shape)


# In[44]:


# 10% vaildation 
x_train,x_test,y_train,y_test = train_test_split(train_imputed,train.loc[:, 'Churn Category'],test_size=0.90, random_state=0)


# ### SVM_linear 

# In[57]:


svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, y_train)
svm_predictions = svm_model_linear.predict(x_test)
accuracy = svm_model_linear.score(x_test, y_test)
print(accuracy)


# ### decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 1).fit(x_train, y_train)
dtree_predictions = dtree_model.predict(x_test)
accuracy = dtree_model.score(x_test, y_test)
print(accuracy)


# ### knn 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 38).fit(x_train, y_train)
accuracy = knn.score(x_test, y_test)
print(accuracy)


# ### RandomForest + ada boost

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 0,n_jobs=-1)
ada = AdaBoostClassifier(n_estimators = 1000, random_state = 0, )



rf.fit(x_train, y_train)
rf.score(x_train, y_train)

#ada.fit(x_train, y_train)
#ada.score(x_train, y_train)


# In[ ]:


rf.score(x_test, y_test)


# ### Prediction

# In[ ]:


testID=pd.read_csv('Test_IDs.csv')
testID.columns=['Customer ID']
dftest=pd.DataFrame(testID)
test=pd.merge(dftest,users,on='Customer ID',how='left')
#print(test)
test_imputed = imputer.fit_transform(test.loc[:, features])
df=pd.DataFrame(test_imputed)
poly = PolynomialFeatures(degree=2, interaction_only=True)
test_imputed = pd.DataFrame(poly.fit_transform(df))
#print(train_imputed.shape)
print(test_imputed)
dftest['Churn Category']=svm_model_linear.predict(test_imputed)

#dftest['Churn Category']=dtree_model.predict(test_imputed)

#testID['Churn Category']=knn.predict(test_imputed)

#dftest['Churn Category']=rfr.predict(test_imputed)
#dftest['Churn Category']=rf.predict(test_imputed)
#dftest['Churn Category']=ada.predict(test_imputed)


# ### Output result

# In[ ]:


testID.columns=['Customer ID','Churn Category']
testID['ans'] = testID['Churn Category']
testID.loc[testID.ans=='No Churn','ans']='0'
testID.loc[testID.ans=='Competitor','ans']='1'
testID.loc[testID.ans=='Dissatisfaction','ans']='2'
testID.loc[testID.ans=='Attitude','ans']='3'
testID.loc[testID.ans=='Price','ans']='4'
testID.loc[testID.ans=='Other','ans']='5'
testID.drop(['Churn Category'],axis=1,inplace=True)
testID.rename(columns={'ans':'Churn Category'}, inplace=True)
submiss=pd.DataFrame(testID)
submiss.to_csv('submission.csv',index=False)

