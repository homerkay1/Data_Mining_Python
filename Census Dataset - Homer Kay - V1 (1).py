
# coding: utf-8

# # - Census Dataset - Homer Kay

# In[2]:


#Add packages
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#From Scikit Learn
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


# In[1]:


# Checking Current Directory


# In[2]:


# Change Directory


# In[5]:


# Importing Data
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
adult = pd.read_csv(url, header=None, na_values=['?'])
adult.head(10)
url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
# Skip First Row, causes encoding problems.  
adult_test = pd.read_csv(url2, header=None, na_values=['?'], skiprows=1)
adult_test.head(10)


# # EDA
# - Rename Columns
# - Combine Training/Test Data
# - Missing Values Columns 10,11?
# - Designate Target Variable
# - Dummies

# In[6]:


# Combining Test/Train From Website
# Decided to do this to perform EDA on entire set and train/test my own sets.  
frames = [adult, adult_test]
adult_comb = pd.concat(frames)


# In[7]:


print(adult_comb.shape)
print(adult_comb.dtypes)
print(adult_comb.mean())
print()
print(adult_comb.std())


# In[8]:


# Rename and Count Missing Values in Columns
adult_comb.columns = ['age', 'workclass','fnlwgt', 'Education','Education#','Marital Status','Occupation','Relationship','Race','Sex','CapGain','CapLoss','Hours per week','Native Country','50K?']
#Check 0's in columns Capital Gains and Capital Losses
print('CapGain # of Non-Zeros')
print(adult_comb.CapGain.astype(bool).sum(axis=0))
print('CapLoss # of Non-Zeros')
print(adult_comb.CapLoss.astype(bool).sum(axis=0))
print('Total # of Columns')
print(adult_comb.CapGain.count())


# ### Considering the fact that CapGain and CapLoss are both zero over 90% of the time; they are missing information, and cannot be imputed.

# In[9]:


#Remove Columns CapGain, CapLoss
adult_c_red = adult_comb.drop(columns=['CapGain','CapLoss'])
adult_c_red.head(10)


# In[10]:


#Designate Target Variable 50K?
targetName = '50K?'
targetSeries = adult_c_red[targetName]
#remove target from current location and insert in column 0
del adult_c_red[targetName]
adult_c_red.insert(0, targetName, targetSeries)
#reprint dataframe and see target is in position 0
adult_c_red.head(10)


# In[11]:


# Replace ? with NA and Remove NA Rows
adult_na = adult_c_red.replace({' ?': np.nan})
adult_red2 = adult_na.dropna()
adult_red2.head(30)
##Confirmed Missing Value Rows are Gone
adult_red2.shape


# ### 45,222 observations left after munging.  
# 

# In[12]:


#Correlation Analysis on Numerical Values
# Numerical
print(adult_red2.corr())
# Visual
plt.matshow(adult_red2.corr())
# None of the variables show particularly high correlation.  


# ### Since Education (Num) and Education (Cat) are essentially duplicates, I just decided to drop Categorical version.  

# In[13]:


# Validating Contents of Numerical and Categorical Counts in Education Variable are Identical.  
print(adult_red2['Education'].value_counts())
print(adult_red2['Education#'].value_counts())


# In[93]:


#Making Dummies
Workclass_d = pd.DataFrame(pd.get_dummies(adult_red2['workclass'])) 
Marital_d = pd.DataFrame(pd.get_dummies(adult_red2['Marital Status']))
Occupation_d = pd.DataFrame(pd.get_dummies(adult_red2['Occupation']))
Relationship_d = pd.DataFrame(pd.get_dummies(adult_red2['Relationship']))
Race_d = pd.DataFrame(pd.get_dummies(adult_red2['Race']))
Sex_d = pd.DataFrame(pd.get_dummies(adult_red2['Sex']))
NativeCountry_d = pd.DataFrame(pd.get_dummies(adult_red2['Native Country']))
fiftyK_d = pd.DataFrame(pd.get_dummies(adult_red2['50K?'])) 
print(fiftyK_d.head(10))


# In[15]:


f1 = fiftyK_d[' <=50K'].sum()
f2 = fiftyK_d[' <=50K.'].sum()
f3 = fiftyK_d[' >50K'].sum()
f4 = fiftyK_d[' >50K.'].sum()
print (f1)
print (f2)
print (f3)
print (f4)


# In[16]:


# Combining Columns
fiftyK_d['<=50K'] = fiftyK_d[' <=50K'] + fiftyK_d[' <=50K.']
f5 = fiftyK_d['<=50K'].sum()
print (f5)
fiftyK_d['>50K'] = fiftyK_d[' >50K'] + fiftyK_d[' >50K.']
f6 = fiftyK_d['>50K'].sum()
print (f6)


# ### 75% of target class if biased towards <=50K.  Dataset is unbalanced.  

# In[17]:


#Create DataFrames of the New Columns
LessEq_50 = pd.DataFrame(fiftyK_d['<=50K'])
Great_50 = pd.DataFrame(fiftyK_d['>50K'])


# In[18]:


#Combining Dummies into Dataframe
result = pd.concat([Great_50, adult_red2, Workclass_d, Marital_d, Occupation_d, Relationship_d, Race_d, Sex_d, NativeCountry_d], axis=1)
#Remove Categorical Columns
result_red = result.drop(columns=['50K?','workclass','Marital Status','Occupation','Relationship','Race','Sex','Native Country','Education'])
print(result_red.head(10))
print(result_red.shape)


# In[19]:


#Viewing all variables to validate none are duplicates/misnomers. 
pd.set_option('display.max_rows', 500) #Display all variables
result_red.dtypes


# ## Categorical EDA

# In[20]:


#weight histogram
n, bins, patches = plt.hist(result_red.fnlwgt, 40, facecolor='blue', alpha=0.75)
plt.ylim(0, 10000)
plt.xlim(0, 500000)
plt.title('histogram of Final Weight')
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()


# In[21]:


#education histogram
Ed = result_red['Education#']
n, bins, patches = plt.hist(Ed, 20, facecolor='red', alpha=0.7)
plt.ylim(0, 15000)
plt.title('histogram of Education')
plt.xlabel("Education Level")
plt.ylabel("Frequency")
plt.show()
# Education Shows peaks at HS-Grad, Some College, and Bachelors.  


# In[22]:


#Hours per week histogram
Hrs = result_red['Hours per week']
n, bins, patches = plt.hist(Hrs, 20, facecolor='green', alpha=0.7)
plt.ylim(0, 30000)
plt.title('histogram of Hours/Week')
plt.xlabel("Hours/Week")
plt.ylabel("Frequency")
plt.show()
# Clearly almost everyone entered 40 hrs/week


# In[23]:


#age histogram
n, bins, patches = plt.hist(result_red.age, 20, facecolor='green', alpha=0.7)
plt.ylim(0, 6000)
plt.title('histogram of Age')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[24]:


# Race 
ra=pd.DataFrame(adult_red2.groupby('Race').size())
ra=ra.sort_values(by=[0])
print(ra.head(10))
print(ra.tail(10))
ra.plot(kind='bar')
#plt.ylim(0, 1000)
plt.ylabel('Frequency')


# In[25]:


# Sex 
sx=pd.DataFrame(adult_red2.groupby('Sex').size())
sx=sx.sort_values(by=[0])
print(sx.head(10))
sx.plot(kind='bar')
#plt.ylim(0, 1000)
plt.ylabel('Frequency')


# In[26]:


# Relationship 
re=pd.DataFrame(adult_red2.groupby('Relationship').size())
re=re.sort_values(by=[0])
print(re.head(10))
re.plot(kind='bar')
#plt.ylim(0, 1000)
plt.ylabel('Frequency')


# In[27]:


# Occupation
oc=pd.DataFrame(adult_red2.groupby('Occupation').size())
oc=oc.sort_values(by=[0])
print(oc.head(10))
print(oc.tail(10))
oc.plot(kind='bar')
#plt.ylim(0, 1000)
plt.ylabel('Frequency')


# In[28]:


# Marital Status 
ms=pd.DataFrame(adult_red2.groupby('Marital Status').size())
ms=ms.sort_values(by=[0])
print(ms.head(10))
ms.plot(kind='bar')
plt.ylabel('Frequency')


# In[29]:


#'workclass','Marital Status','Occupation','Relationship','Race','Sex
# Workclass 
wc=pd.DataFrame(adult_red2.groupby('workclass').size())
wc=wc.sort_values(by=[0])
print(wc.head(10))
print(wc.tail(10))
wc.plot(kind='bar')
#plt.ylim(0, 1000)
plt.ylabel('Frequency')


# In[30]:


# Country 
ctry_income=pd.DataFrame(adult_red2.groupby('Native Country').size())
ctry_income=ctry_income.sort_values(by=[0])
print(ctry_income.head(10))
print(ctry_income.tail(10))
ctry_income.plot(kind='bar')
plt.ylim(0, 1000)
plt.ylabel('Frequency')
# Over 90% of Rows come from United States
# Limited Y to show countries other than US


# ## Test/Train

# In[133]:


#Train/Test Split (70/30 since dataset is large)
features_train, features_test, target_train, target_test = train_test_split(
    result_red.iloc[:,1:].values, result_red.iloc[:,0].values, test_size=0.30, random_state=0)
print(features_test.shape)
print(features_train.shape)
print(target_test.shape)
print(target_train.shape)


# In[75]:


import numpy as np
from sklearn.decomposition import PCA
X = ft_red
pca = PCA(n_components=20)
pca1 = pca.fit(X)
pca2 = PCA(copy=True, iterated_power='auto', n_components=20, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  


# In[77]:


fit1 = pca.fit_transform(X, y=None)


# In[83]:


fit_ft_red = pd.DataFrame(fit1)


# # Models

# ## KNN

# In[153]:


## KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh_knn = neigh.fit(features_train, target_train) 
target_predicted_knn = neigh_knn.predict(features_test)


# ### Grid Search KNN = 1-15

# In[150]:


# use a full grid over several parameters and cross validate 5 times
from sklearn.model_selection import GridSearchCV
#param_grid = {"alpha": [.01,.1, .5, 1, 2]}
param_grid={"n_neighbors": [10,15,1]} #this does a range 1 through 10 changes by a factor of 1. 
#param_grid={"n_neighbors": [10,15,1]} #this does a range 1 through 1 changes by a factor of .05

# run grid search
grid_search = GridSearchCV(neigh, param_grid=param_grid,n_jobs=-1,cv=5)
grid_search.fit(features_train, target_train)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)   


# In[154]:


print("KNN Accuracy Score", accuracy_score(target_test, target_predicted_knn))
print("Classification Report")
print(classification_report(target_test, target_predicted_knn))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_knn))


# In[155]:


#verify KNN with Cross Validation for KNN = 10
scores = cross_val_score(neigh_knn, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean() 


# ### Running KNN through 1-10, and then through 10-15 resulted in most optimized model of KNN=10.

# ## Random Forest

# In[185]:


# Random Forest N=10
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators = 10)
rf_fit = rf.fit(features_train, target_train)
target_predicted_rf = rf_fit.predict(features_test)
print("RF Accuracy Score N=10", accuracy_score(target_test, target_predicted_rf))
print("Classification Report")
print(classification_report(target_test, target_predicted_rf))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_rf))


# In[182]:


# Random Forest N=100
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators = 100)
rf_fit = rf.fit(features_train, target_train)
target_predicted_rf = rf_fit.predict(features_test)
print("RF Accuracy Score N=100", accuracy_score(target_test, target_predicted_rf))
print("Classification Report")
print(classification_report(target_test, target_predicted_rf))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_rf))


# In[183]:


# Random Forest N=500
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators = 500)
rf_fit = rf.fit(features_train, target_train)
target_predicted_rf = rf_fit.predict(features_test)
print("RF Accuracy Score N=500", accuracy_score(target_test, target_predicted_rf))
print("Classification Report")
print(classification_report(target_test, target_predicted_rf))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_rf))


# In[186]:


#verify RF with Cross Validation on N=10
scores = cross_val_score(rf_fit, features_train, target_train, cv=5)
print("Cross Validation Score",scores)
scores.mean() 


# ## Decision Tree

# In[187]:


## Decision Tree Max Depth = 5
from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier(max_depth=5, class_weight="balanced")
#Call up the model to see the parameters you can tune (and their default setting)
print(clf_dt)
#Fit clf to the training data
clf_dt = clf_dt.fit(features_train, target_train)
#Predict clf DT model again test data
target_predicted_dt = clf_dt.predict(features_test)
print("DT Accuracy Score Max Depth = 5", accuracy_score(target_test, target_predicted_dt))
print("Classification Report")
print(classification_report(target_test, target_predicted_dt))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_dt))


# In[190]:


## Decision Tree Max Depth = 15
from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier(max_depth=15, class_weight="balanced")
#Call up the model to see the parameters you can tune (and their default setting)
print(clf_dt)
#Fit clf to the training data
clf_dt = clf_dt.fit(features_train, target_train)
#Predict clf DT model again test data
target_predicted_dt = clf_dt.predict(features_test)
print("DT Accuracy Score Max Depth = 15", accuracy_score(target_test, target_predicted_dt))
print("Classification Report")
print(classification_report(target_test, target_predicted_dt))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_dt))


# In[189]:


## Decision Tree Max Depth = none
from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier(class_weight="balanced")
#Call up the model to see the parameters you can tune (and their default setting)
print(clf_dt)
#Fit clf to the training data
clf_dt = clf_dt.fit(features_train, target_train)
#Predict clf DT model again test data
target_predicted_dt = clf_dt.predict(features_test)
print("DT Accuracy Score Max Depth = none", accuracy_score(target_test, target_predicted_dt))
print("Classification Report")
print(classification_report(target_test, target_predicted_dt))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_dt))


# In[191]:


#verify DT with Cross Validation for Max Depth = 15
scores = cross_val_score(clf_dt, features_train, target_train, cv=5)
print("Cross Validation Score",scores)
scores.mean()  


# ## SVM Linear

# In[141]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(features_train)
# Now apply the transformations to the data:
X_train = scaler.transform(features_train)
X_test = scaler.transform(features_test)


# In[142]:


from sklearn.svm import LinearSVC
clf_linSVC=LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=0.1, class_weight='balanced', max_iter = 100)
clf_linSVC.fit(X_train, target_train)
predicted_SVC=clf_linSVC.predict(X_test)
expected = target_test
# summarize the fit of the model
print('C=0.1')
print(classification_report(expected, predicted_SVC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVC))
print(accuracy_score(expected,predicted_SVC))


# In[145]:


from sklearn.svm import LinearSVC
clf_linSVC=LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=0.5, class_weight='balanced', max_iter = 100)
clf_linSVC.fit(X_train, target_train)
predicted_SVC=clf_linSVC.predict(X_test)
expected = target_test
# summarize the fit of the model
print('C=0.5')
print(classification_report(expected, predicted_SVC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVC))
print(accuracy_score(expected,predicted_SVC))


# In[144]:


from sklearn.svm import LinearSVC
clf_linSVC=LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=1.0, class_weight='balanced', max_iter = 100)
clf_linSVC.fit(X_train, target_train)
predicted_SVC=clf_linSVC.predict(X_test)
expected = target_test
# summarize the fit of the model
print('C=1.0')
print(classification_report(expected, predicted_SVC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVC))
print(accuracy_score(expected,predicted_SVC))


# In[146]:


#verify SVM Linear with Cross Validation C = 0.5
scores = cross_val_score(clf_linSVC, X_train, target_train, cv=10)
print("Cross Validation Score",scores)
scores.mean()  


# ### SVM Linear, 76% Accuracy, Verified with Cross Validation.  

# ## Extra Trees

# In[213]:


from sklearn.ensemble import ExtraTreesClassifier
xdt = ExtraTreesClassifier(max_depth=3,
                         n_estimators=10,class_weight='balanced')
#print(xdt)
xdt2=xdt.fit(features_train, target_train)
predicted_xdt=xdt.predict(features_test)
expected = target_test
print("Extra Trees", accuracy_score(expected,predicted_xdt))
print(classification_report(expected, predicted_xdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_xdt))


# In[214]:


from sklearn.ensemble import ExtraTreesClassifier
xdt = ExtraTreesClassifier(max_depth=10,
                         n_estimators=10,class_weight='balanced')
#print(xdt)
xdt2=xdt.fit(features_train, target_train)
predicted_xdt=xdt.predict(features_test)
expected = target_test
print("Extra Trees", accuracy_score(expected,predicted_xdt))
print(classification_report(expected, predicted_xdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_xdt))


# In[215]:


from sklearn.ensemble import ExtraTreesClassifier
xdt = ExtraTreesClassifier(
                         n_estimators=10,class_weight='balanced')
#print(xdt)
xdt2=xdt.fit(features_train, target_train)
predicted_xdt=xdt.predict(features_test)
expected = target_test
print("Extra Trees", accuracy_score(expected,predicted_xdt))
print(classification_report(expected, predicted_xdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_xdt))


# In[216]:


from sklearn.ensemble import ExtraTreesClassifier
xdt = ExtraTreesClassifier(
                         n_estimators=100,class_weight='balanced')
#print(xdt)
xdt2=xdt.fit(features_train, target_train)
predicted_xdt=xdt.predict(features_test)
expected = target_test
print("Extra Trees", accuracy_score(expected,predicted_xdt))
print(classification_report(expected, predicted_xdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_xdt))


# In[217]:


#verify Extra Trees with Cross Validation max_depth = none, n_estimators = 100
scores = cross_val_score(xdt2, features_train, target_train, cv=10)
print("Cross Validation Score",scores)
scores.mean()  


# ### Extra Trees improves from 71% to 80% from removing max depth parameter.  Slight improvement ~1% seen from increasing estimators from 10->100.  CV Confirms validity of model.  OK to productionalize.  

# ## Gradient Boosting

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.7, max_depth=1, random_state=0)
clf_GBC.fit(features_train, target_train)
predicted_GBC=clf_GBC.predict(features_test)
expected = target_test
print("Gradient Boost Accuracy", accuracy_score(expected,predicted_GBC))
print(classification_report(expected, predicted_GBC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_GBC))


# In[219]:


from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, max_depth=1, random_state=0)
clf_GBC.fit(features_train, target_train)
predicted_GBC=clf_GBC.predict(features_test)
expected = target_test
print("Gradient Boost Accuracy", accuracy_score(expected,predicted_GBC))
print(classification_report(expected, predicted_GBC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_GBC))


# In[226]:


from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=10, learning_rate=0.6, max_depth=1, random_state=0)
clf_GBC.fit(features_train, target_train)
predicted_GBC=clf_GBC.predict(features_test)
expected = target_test
print("Gradient Boost Accuracy", accuracy_score(expected,predicted_GBC))
print(classification_report(expected, predicted_GBC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_GBC))


# In[229]:


from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=10, learning_rate=0.6, random_state=0)
clf_GBC.fit(features_train, target_train)
predicted_GBC=clf_GBC.predict(features_test)
expected = target_test
print("Gradient Boost Accuracy", accuracy_score(expected,predicted_GBC))
print(classification_report(expected, predicted_GBC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_GBC))


# In[230]:


from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=10, learning_rate=0.6, random_state=0, loss='exponential')
#loss : {‘deviance’, ‘exponential’}, optional (default=’deviance’)
clf_GBC.fit(features_train, target_train)
predicted_GBC=clf_GBC.predict(features_test)
expected = target_test
print("Gradient Boost Accuracy", accuracy_score(expected,predicted_GBC))
print(classification_report(expected, predicted_GBC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_GBC))


# In[231]:


#verify Gradient Boosting with Cross Validation 
scores = cross_val_score(clf_GBC, features_train, target_train, cv=10)
print("Cross Validation Score",scores)
scores.mean()  


# ### Best Gradient Boosting, Learning Rate 0.6, Exponential Loss, No Max Depth.  High Accuracy.  CV Passes.  Could Productionalize.  

# ## Adaboost

# In[232]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[233]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[235]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),
                         algorithm="SAMME",
                         n_estimators=20)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[239]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(
                         algorithm="SAMME",
                         n_estimators=20)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[240]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[241]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20, learning_rate=2)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[242]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20, learning_rate=1)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[244]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20, learning_rate=0.7)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[252]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
bdt = AdaBoostClassifier(RandomForestClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20, learning_rate=0.7)
bdt.fit(features_train, target_train)
predicted_bdt=bdt.predict(features_test)
expected = target_test
print("Adaboost Accuracy", accuracy_score(expected,predicted_bdt))
print(classification_report(expected, predicted_bdt,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bdt))


# In[245]:


#verify AdaBoost with Cross Validation 
scores = cross_val_score(bdt, features_train, target_train, cv=10)
print("Cross Validation Score",scores)
scores.mean()  


# ### Best Model, Decision Tree Base Learner, Learning Rate = 1, Estimators = 20, Max Depth = 3
# 
# #### Little to no difference between Decision Tree and Random Forest.

# # Stacking

# In[180]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#Three Models RF, NB, BDT
clf1 = RandomForestClassifier(random_state=1)
clf_LR = LogisticRegression()
clf2 = GaussianNB()
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200)
eclf2 = VotingClassifier(estimators=[('rf', clf1), ('gnb', clf2), ('bdt', bdt)], voting='hard')
for MV, label in zip([clf1, clf2, bdt, eclf2, clf_LR], ['Random Forest', 'naive Bayes', 'AdaBoost Decision Tree', 'Ensemble','Logistic Regression']):

    scores2 = cross_val_score(MV, features_train, target_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label))


# In[182]:


eclf2.fit(features_train, target_train)
predictions = eclf2.predict(features_test)
from sklearn.metrics import classification_report,confusion_matrix
print("Accuracy", accuracy_score(target_test,predictions))
print(classification_report(target_test,predictions))
print(confusion_matrix(target_test,predictions))


# In[183]:


#verify Stacking with Cross Validation 
scores = cross_val_score(eclf2, features_train, target_train, cv=3)
print("Cross Validation Score",scores)
scores.mean()  


# ### Stacking 83% Accuracy, CV Verified

# # Bagging

# In[255]:


#Bagging Classifer
from sklearn.ensemble import BaggingClassifier # base estimator is base learner algorithm.  
from sklearn.neighbors import KNeighborsClassifier
clf_bag = BaggingClassifier(KNeighborsClassifier(), max_samples = 0.5, max_features=0.5)
print(clf_bag)
clf_bag.fit(features_train, target_train)
predicted_bag=clf_bag.predict(features_test)
expected = target_test
print("Bagging Accuracy", accuracy_score(expected,predicted_bag))
print(classification_report(expected, predicted_bag,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bag))


# In[261]:


#Bagging Classifer
from sklearn.ensemble import BaggingClassifier # base estimator is base learner algorithm.  I.E. is you say Decision tree you are making a random forest. 
clf_bag = BaggingClassifier(n_estimators=10, random_state=0)
print(clf_bag)
clf_bag.fit(features_train, target_train)
predicted_bag=clf_bag.predict(features_test)
expected = target_test
print("Bagging Accuracy", accuracy_score(expected,predicted_bag))
print(classification_report(expected, predicted_bag,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bag))


# In[257]:


#Bagging Classifer
from sklearn.ensemble import BaggingClassifier # base estimator is base learner algorithm.  
from sklearn.linear_model import LogisticRegression
clf_bag = BaggingClassifier(LogisticRegression(), max_samples = 0.5, max_features=0.5)
print(clf_bag)
clf_bag.fit(features_train, target_train)
predicted_bag=clf_bag.predict(features_test)
expected = target_test
print("Bagging Accuracy", accuracy_score(expected,predicted_bag))
print(classification_report(expected, predicted_bag,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_bag))


# In[262]:


#verify Bagging with Cross Validation #DT Method
scores = cross_val_score(clf_bag, features_train, target_train, cv=10)
print("Cross Validation Score",scores)
scores.mean()  


# ### Best Bagging method used DT Base Learner, with Default Variables.  CV Verifies it is Ready for Production.  

# # SVM RBF (Takes a long time to run)

# In[167]:


# Reducing Rows for RBF
ft = pd.DataFrame(features_train)
ft_red = ft[:5000]
ft_red.shape
tt = pd.DataFrame(target_train)
tt_red = tt[:5000]
tt_red.shape
ftest = pd.DataFrame(features_test)
ftest_red = ftest[:5000]
ftest_red.shape
#TIME NOTES
# 1000 Rows = 1.7 Sec
# 5000 Rows = 8.5 Sec
# 20,000 Rows = 198 Sec
# Settled on 5000 Rows for the sake of sanity.  
# POLY at 5,000 takes...


# In[168]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(ft_red)
# Now apply the transformations to the data:
X_train_RBF = scaler.transform(ft_red)
X_test_RBF = scaler.transform(ftest_red)


# In[169]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import svm, datasets
import time
start_time = time.clock()
C = 1.0  # SVM regularization parameter
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train_RBF, tt_red)
predicted_svm_rbf = rbf_svc.predict(X_test_RBF)
expected = tt_red
print("SVM RBF Accuracy", accuracy_score(expected,predicted_svm_rbf))
print(classification_report(expected, predicted_svm_rbf,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_svm_rbf))
print(time.clock() - start_time, "seconds")


# In[170]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import svm, datasets
import time
start_time = time.clock()
C = 1.0  # SVM regularization parameter
rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=C).fit(X_train_RBF, tt_red)
predicted_svm_rbf = rbf_svc.predict(X_test_RBF)
expected = tt_red
print("SVM RBF Accuracy", accuracy_score(expected,predicted_svm_rbf))
print(classification_report(expected, predicted_svm_rbf,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_svm_rbf))
print(time.clock() - start_time, "seconds")


# In[178]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import svm, datasets
import time
start_time = time.clock()
C = 0.5  # SVM regularization parameter
rbf_svc = svm.SVC(kernel='rbf', gamma=0.3, C=C).fit(X_train_RBF, tt_red)
predicted_svm_rbf = rbf_svc.predict(X_test_RBF)
expected = tt_red
print("SVM RBF Accuracy", accuracy_score(expected,predicted_svm_rbf))
print(classification_report(expected, predicted_svm_rbf,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_svm_rbf))
print(time.clock() - start_time, "seconds")


# In[177]:


#verify SVM RBF with Cross Validation C=0.5 gamma=0.3
scores = cross_val_score(rbf_svc, X_train_RBF, tt_red, cv=5)
print("Cross Validation Score",scores)
scores.mean()  


# ### SVM RBF - Best accuracy 76%, Verified with CV

# # Stochastic Gradient Descent

# In[147]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(features_train)
# Now apply the transformations to the data:
X_train = scaler.transform(features_train)
X_test = scaler.transform(features_test)


# In[148]:


## SGD
from sklearn.linear_model import SGDClassifier 
clf_sgd = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf_sgd = clf_sgd.fit(X_train, target_train)
target_predicted_sgd = clf_sgd.predict(X_test)
print("SGD Accuracy Score", accuracy_score(target_test, target_predicted_sgd))
print("Classification Report")
print(classification_report(target_test, target_predicted_sgd))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_sgd))


# In[153]:


## SGD
from sklearn.linear_model import SGDClassifier 
clf_sgd = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.5,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf_sgd = clf_sgd.fit(X_train, target_train)
target_predicted_sgd = clf_sgd.predict(X_test)
print("SGD Accuracy Score", accuracy_score(target_test, target_predicted_sgd))
print("Classification Report")
print(classification_report(target_test, target_predicted_sgd))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_sgd))


# In[150]:


## SGD
from sklearn.linear_model import SGDClassifier 
clf_sgd = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.9, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf_sgd = clf_sgd.fit(X_train, target_train)
target_predicted_sgd = clf_sgd.predict(X_test)
print("SGD Accuracy Score", accuracy_score(target_test, target_predicted_sgd))
print("Classification Report")
print(classification_report(target_test, target_predicted_sgd))
print("Confusion Matrix")
print(confusion_matrix(target_test, target_predicted_sgd))


# In[155]:


#verify SGD with Cross Validation
scores = cross_val_score(clf_sgd, X_train, target_train, cv=10)
print("Cross Validation Score",scores)
scores.mean()  


# ### SGD 79% Accurate, CV Verified.  

# # Artificial Neural Network

# In[134]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(features_train)
# Now apply the transformations to the data:
X_train = scaler.transform(features_train)
X_test = scaler.transform(features_test)


# In[138]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,5))
print(mlp)
mlp.fit(X_train,target_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("Accuracy", accuracy_score(target_test,predictions))
print(classification_report(target_test,predictions))
print(confusion_matrix(target_test,predictions))


# In[136]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,10,5))
print(mlp)
mlp.fit(X_train,target_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("Accuracy", accuracy_score(target_test,predictions))
print(classification_report(target_test,predictions))
print(confusion_matrix(target_test,predictions))


# In[137]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,10))
print(mlp)
mlp.fit(X_train,target_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("Accuracy", accuracy_score(target_test,predictions))
print(classification_report(target_test,predictions))
print(confusion_matrix(target_test,predictions))


# In[140]:


#verify ANN with Cross Validation Hidden 10,5 layers
scores = cross_val_score(mlp, X_train, target_train, cv=5)
print("Cross Validation Score",scores)
scores.mean()  


# ### ANN Accuracy 83%, CV Verified
