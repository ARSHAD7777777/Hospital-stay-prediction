#!/usr/bin/env python
# coding: utf-8

# # CAPSTONE PROJECT 4
# ## PROJECT NAME: 
# ### Hospital Stay Prediction
# 
# ## PROJECT CODE:
# ### PRCP-1022-HospitalStayPred
# 
# ## PROJECT TEAM ID:
# ### PTID-CDS-JUL21-1172															

# OBJECTIVE:
# Using the featutres given in the dataset to predict the number of days a patient can stay after he visits the hospital

# ### INTRODUCTION
# The given dataset presents the information regarding patients visiting the hospital.It indicates the number of days a particular patient stays in the hospital along with the values of other features including:
# #####   case_id                              
# #####   Hospital_code                        
# #####   Hospital_type_code                  
# #####   City_Code_Hospital                   
# #####   Hospital_region_code                
# #####   Available_Extra_Rooms_in_Hospital   
# #####   Department                          
# #####   Ward_Type                          
# #####   Ward_Facility_Code                  
# #####   Bed_Grade                          
# #####   patientid                            
# #####   City_Code_Patient                  
# #####   Type_of_Admission                  
# #####   Severity_of_Illness                 
# #####   Visitors_with_Patient               
# #####   Age                                 
# #####   Admission_Deposit                    
# #####   Stay               
# The aim of this project is to build a machine learning model that can predict the number of days a patient can stay in the hospital given the values of the other relevant features

# ## Dataset Description:
# Majority of the features given in the dataset are categorical variables having a set of unique values. The target variable is given as 'Stay' the values of which are given as the following intervals:
# 
# 
# ['0-10','41-50','31-40','21-30',51-60','71-80','81-90','61-70','91-100','More than 100 Days']
# 
# Therefore the Project is treated as a multi-class classification problem.

# ## STEP 1: Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import RandomizedSearchCV


# ## Step 2: Loading the Dataset

# In[2]:


pwd


# In[3]:


cd "C:\Users\arshad\Downloads\HealthCareAnalytics"


# In[4]:


data= pd.read_csv('HealthCareAnalytics.csv')


# ## Step 3: Exploratory Data Analysis(EDA)

# In[5]:


data.head()


# In[6]:


data.info()


# #### Checking for null values
# The features "Age" and "Stay" has got the value 'Nov-20' in some samples. As this value seems irrelevant in comparison with other values of the given features, it is considered a typo and treated as null values.

# In[7]:


data['Age']=data['Age'].replace('Nov-20',np.nan)


# In[8]:


data['Stay']=data['Stay'].replace('Nov-20',np.nan)


# In[9]:


data.isnull().sum()


# #### Checking for auto_correlation using pairplot from seaborn

# In[48]:


sns.pairplot(data,hue='Stay')


# From the above pairplot it can be understood that none of the features show strong correlations.

# #### Checking for Imbalance in Target variable values

# In[9]:


plt.figure(figsize=(15,8))
sns.countplot(data['Stay'])


# The target variable values in the given dataset are imbalanced

# #### Checking the unique values in categorical variables

# In[15]:


for i in data.columns:
    print(i,'unique values')
    print(data[i].unique())


# ## Step 3: Data preprocessing

# #### Dropping the irrelevant variables 

# In[5]:


data.drop(['case_id','patientid'],axis=1,inplace=True)


# #### Dealing with the null values using SimpleImputer from scikitlearn

# In[6]:


imputer=SimpleImputer(strategy='most_frequent')
null_list=['Bed_Grade','Age','Stay','City_Code_Patient']
for i in null_list:
    data[i]=imputer.fit_transform(data[[i]])


# In[7]:


data.isnull().sum()


# #### Inorder to get unique features while one-hot encoding, some variables with similar categorical values are mapped into different values.

# In[16]:


h_list=[]
for i in range(32):
    h_list.append('h'+str(i))
h_dict=dict(zip(sorted(data['Hospital_code'].unique()),h_list))


# In[17]:


h_dict


# In[18]:


data['Hospital_code']=data['Hospital_code'].astype('object')


# In[19]:


data['Hospital_code']=data['Hospital_code'].map(h_dict)


# In[20]:


data.head()


# In[21]:


data['City_Code_Hospital']=data['City_Code_Hospital'].astype('object')


# In[22]:


c_list=[]
for i in range(len(data['City_Code_Hospital'].unique())):
    c_list.append('c'+str(i))
c_dict=dict(zip(sorted(data['City_Code_Hospital'].unique()),c_list))
c_dict


# In[23]:


data['City_Code_Hospital']=data['City_Code_Hospital'].map(c_dict)


# In[24]:


data.head()


# In[25]:


data.info()


# #### Splitting the predictor and target variables into x and y respectively.

# In[26]:


x=data.iloc[:,:15]
y=data.iloc[:,15]


# In[27]:


x.head()


# In[28]:


y.head()


# #### One-hot encoding the categorical variables after excluding the discrete quantitative variables and forming a new dataframe with both.

# In[10]:


encoder=OneHotEncoder(sparse=False)


# In[29]:


int_features=[]
for i in x.columns:
    if x[i].dtype=='int64':
        int_features.append(i)
int_features


# In[30]:


x_new=x[int_features]
x_new.head()


# In[34]:


for i in x.columns:
    if x[i].dtype=='object' or x[i].dtype=='float64':
        encoded_feature_df=pd.DataFrame(encoder.fit_transform(x[[i]]),columns=x[i].unique())
        x_new=x_new.join(encoded_feature_df,how='left',lsuffix='l')


# In[35]:


x_new.head()


# #### Balancing the target variable values using the SMOTE function from imblearn API

# In[37]:


oversampler=SMOTE()
x_new,y_new=oversampler.fit_resample(x_new,y)


# In[38]:


sns.countplot(y_new)


# #### Encoding the target variable

# In[39]:


yencoder=LabelEncoder()


# In[46]:


y_new=yencoder.fit_transform(y_new)


# In[47]:


y_new


# #### Splitting the preprocessed dataset into training and test data

# In[48]:


x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.20, random_state=20)


# In[49]:


y_train


# In[50]:


x_train


# #### Trying out the various classification models and artificial neural network and selecting the best one for the problem using accuracy score

# ###### Machine learning models

# In[11]:


knn=KNeighborsClassifier(n_neighbors=3)
forest=RandomForestClassifier(max_depth=4)
adab=AdaBoostClassifier()
xgb=XGBClassifier(max_depth=4)


# In[38]:


knn.fit(x_train,y_train)


# In[39]:


xgb.fit(x_train,y_train)


# In[40]:


forest.fit(x_train,y_train)


# In[41]:


adab.fit(x_train,y_train)


# In[44]:


model_list=[adab,forest,xgb,knn]
for i in model_list:
    print('accuracy score for',i,'model is')
    print(accuracy_score(y_true=y_test,y_pred=i.predict(x_test)))


# ##### Artificial neural network

# In[39]:


model=Sequential()


# In[70]:


data['Stay'].unique().shape


# In[40]:


model.add(Dense(500,activation='relu'))


# In[41]:


model.add(Dense(250,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[42]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')


# In[43]:


model.fit(x_train,y_train,epochs=10,batch_size=10)


# Based on the results it can be understood that the XGBoost classifier best suits the classification problem in hand.

# #### Using RandomizedSearchCV from scikit learn inorder to carry out the Hyperparameter tuning of  the XGBoost model.

# In[46]:


parameters={'max_depth':[2,3,4,5],'n_estimators':[200,500,750,1000,1500,2000],'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6]}
model=XGBClassifier(enable_categorical=True)
grid=RandomizedSearchCV(model,parameters,cv=3)


# In[47]:


grid.fit(data_new,y)


# In[48]:


grid.best_params_


# In[50]:


model=grid.best_estimator_


# In[52]:


p=model.predict(x_test)


# In[54]:


accuracy_score(y_true=y_test,y_pred=p)


# #### Saving and loading the best model using joblib

# In[56]:


joblib.dump(model,'healthcare_model')


# In[42]:


hosptital_stay_model=joblib.load('healthcare_model')


# ## PROJECT ANALYSIS
# The object of the project was to build a machine learning model which can predict the duration in days a patient stays in a hospital depending upon the values of various features. The given dataset had 18 features including the target variable and 318438 rows or samples. The exploratory data analysis part involved:
# 
# - Checking autocorrelations using pairplot from seaborn API
# - Checking null values
# - Finding unique values of each categorical variable
# - Checking for imbalance in the target variable values
# 
# From the pairplot it was obvious that none of the features were strongly correlated thus ruling out the option of feature elimination. The features "Age" and "Stay" has got the value 'Nov-20' in some samples. As this value seems irrelevant in comparison with other values of the given features, it was considered a typo and treated as null values.The countplot of the target variable revealed the imbalance in values of the same.
# 
# Data preprocessing involved 
# 
# - Dropping irrelevant variables
# - Splitting the dataset into predictor and target variables
# - Imputing the null values with the most frequent value/mode of each feature using the SimpleImputer from scikitlearn
# - Using SMOTE from imblearn API to generate synthetic data inorder to balance the target variable
# - Mapping the values of some categorical variables into other values to get unique features during one hot encoding
# - One-hot encoding the categorical variables after excluding the discrete quantitative variables and forming a new dataframe   with both
# - Encoding the target variable
# - Splitting the preprocessed dataset into training and test data
# 
# The next step was trying out the various machine learning models for multi-class classification. The followin models were studied for their accuracy
# 
# - RandomForestClassifier
# - XGBClassifier
# - AdaboostClassifier
# - KNeighborsClassifier
# 
# Along with this an artificial neural network was also tried out.
# It was observed that the XGBoost classifeir model was more efficient than the other models and was consequently used for the project. The RandomizedSearchCV from scikitlearn was used to carry out the Hyper-parameter tuning of XGBClassifier and the best model was selected, tested with the test data and saved using joblib.
# The complexity of the dataset, particularly after balancing the target variable was a constraint to check the efficiency of more combinations of hyper parameters as it would take a lot of computational power and time.However with the same workflow, much better accuracies can be achieved using a strong processor.
# 
# 
