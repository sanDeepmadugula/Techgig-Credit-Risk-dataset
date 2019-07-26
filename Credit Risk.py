#!/usr/bin/env python
# coding: utf-8

# In[269]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[270]:


import os
os.chdir('C:\\Analytics\\MachineLearning\\credit rist\\Complete-Data-Set')


# In[271]:


test_data = pd.read_csv('application_test.csv')
train_data = pd.read_csv('application_train.csv')


# In[272]:


test_data.sample(10)


# In[273]:


train_data.sample(10)


# In[274]:


train_data.info()


# In[275]:


test_data.info()


# In[276]:


train_data.isnull().sum()


# In[277]:


# lets calculate the percentage of missing values
train_data.count()/len(train_data)


# In[278]:


plt.figure(figsize=(10,8))
sns.countplot('TARGET',data=train_data)


# There is a clear problem here, we have unbalanced target class. will check the event rate.

# In[279]:


class_0 = train_data.TARGET.value_counts()[0]
class_1 = train_data.TARGET.value_counts()[1]
print("Total number of class_0: {}".format(class_0))
print("Total number of class_1: {}".format(class_1))
print("Event rate: {} %".format(class_1/(class_0+class_1) *100))


# We have an event rate of 6.68%, consequences of having this kind of target class is most likely that the minority class is being ignored by the algorithm and will predict the new instances to class_0 as it was the safest way to have a great accuracy.
# Since we have a lot of data will consider Resampling(Under-sampling to be exact) this strategy
# will randomly delete some of the instances of the majority class(class_0)

# Using penalized models(penalized RF, Logit)

# Considering ensemble models

# In[280]:


train_data.columns


# In[213]:


corr = train_data.corr()
plt.figure(figsize=(14,12))
sns.heatmap(corr,annot=True,fmt='.2g')


# In[214]:


def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j] >=threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]
print(train_data)


# In[89]:


corr_matrix = train_data.corr().abs()
high_corr_var=np.where(corr_matrix>0.8)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]


# In[90]:


high_corr_var


# In[281]:


missing_df = train_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig,ax = plt.subplots(figsize=(20,26))
rects = ax.barh(ind, missing_df.missing_count.values, color='red')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values,rotation='horizontal')
ax.set_xlabel('Count of missing values')
ax.set_title('Number of missing values in each column')
plt.show()


# In[282]:


cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]
reduced_original_data = train_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing, axis=1)


# In[283]:


reduced_original_data.shape


# In[284]:


reduced_test_data.shape


# In[149]:


corr = reduced_original_data.corr()
plt.figure(figsize=(14,12))
sns.heatmap(corr,annot=True,fmt='.2g')


# In[285]:


trained = reduced_original_data.copy()
tested = reduced_test_data.copy()


# In[286]:


trained.shape, tested.shape


# In[287]:


trained.columns


# In[288]:


# check which are numerical and which are categorical
trained_num_cols = trained._get_numeric_data().columns
trained_num_cols


# In[289]:


trained.drop([
    'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21'
],axis=1,inplace=True)


# In[290]:


trained.columns


# In[291]:


trained.drop(['FLAG_OWN_REALTY','FLAG_MOBIL','FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
             'FLAG_PHONE', 'FLAG_EMAIL','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY'],axis=1,inplace=True)


# In[292]:


trained.shape


# In[293]:


trained.columns


# In[294]:


tested.drop([
    'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21','FLAG_OWN_REALTY','FLAG_MOBIL','FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
             'FLAG_PHONE', 'FLAG_EMAIL','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY'
],axis=1,inplace=True)


# In[295]:


tested.shape


# In[296]:


trained.shape


# In[297]:


trained.columns


# In[298]:


tested.columns


# In[299]:


trained.sample(5)


# In[300]:


trained.NAME_INCOME_TYPE.unique()


# In[301]:


trained.columns


# In[302]:


trained.shape


# In[303]:


trained.columns


# In[304]:


trained.head()


# In[305]:


trained['NAME_CONTRACT_TYPE'] = trained['NAME_CONTRACT_TYPE'].map({'Cash loans':1,'Revolving loans':2 })
trained['CODE_GENDER'] = trained['CODE_GENDER'].map({'F':1,'M':2,'XNA':3})
trained['FLAG_OWN_CAR'] = trained['FLAG_OWN_CAR'].map({'N':0,'Y':1})
trained['NAME_INCOME_TYPE'] = trained['NAME_INCOME_TYPE'].map({'State servant':1,'Pensioner':2,'Working':3,'Commercial associate':4,'Student':5,'Unemployed' :6,'Businessman':7,'Maternity leave':8})


# In[306]:


trained.head()


# In[307]:


trained.drop(['NAME_EDUCATION_TYPE', 'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','NAME_FAMILY_STATUS' ],axis=1,inplace=True)


# In[308]:


trained.head()


# In[309]:


trained.drop(['NAME_HOUSING_TYPE'],axis=1,inplace=True)


# In[310]:


trained.head()


# In[311]:


trained.shape


# In[312]:


tested.shape


# In[313]:


tested.head()


# In[314]:


tested['NAME_CONTRACT_TYPE'] = tested['NAME_CONTRACT_TYPE'].map({'Cash loans':1,'Revolving loans':2 })
tested['CODE_GENDER'] = tested['CODE_GENDER'].map({'F':1,'M':2,'XNA':3})
tested['FLAG_OWN_CAR'] = tested['FLAG_OWN_CAR'].map({'N':0,'Y':1})
tested['NAME_INCOME_TYPE'] = tested['NAME_INCOME_TYPE'].map({'State servant':1,'Pensioner':2,'Working':3,'Commercial associate':4,'Student':5,'Unemployed' :6,'Businessman':7,'Maternity leave':8})


# In[315]:


tested.head()


# In[317]:


tested.drop(['NAME_EDUCATION_TYPE', 'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE'],axis=1,inplace=True)


# In[318]:


tested.shape


# In[319]:


X = trained.drop("TARGET", axis=1).copy()
y = train_data.TARGET
X.shape, y.shape


# In[320]:


trained.dtypes


# In[193]:





# In[321]:


trained.shape


# In[322]:


tested.shape


# In[ ]:





# In[323]:


tested.shape


# In[324]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler

X_train, X_val, y_train,y_val = train_test_split(X,y,random_state=42)
logit = LogisticRegression(random_state=42, solver='saga', penalty='l1',class_weight='balanced', C=1.0, max_iter=500)
scaler=  StandardScaler().fit(X_train)


# In[325]:


X_trained_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)


# In[326]:


logit.fit(X_trained_scaled, y_train)
logit_scores_proba = logit.predict_proba(X_trained_scaled)
logit_scores = logit_scores_proba[:,1]


# In[327]:


# lets make a roc curve visualization
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(12,10))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[329]:


fpr_logit, tpr_logit, thresh_logit = roc_curve(y_train, logit_scores)
plot_roc_curve(fpr_logit,tpr_logit)
print("AUC Score {}".format(roc_auc_score(y_train,logit_scores)))


# In[330]:


# validate with validation set
logit_scores_proba_val = logit.predict_proba(X_val_scaled)
logit_scores_val = logit_scores_proba_val[:,1]
fpr_logit_val, tpr_logit_val, thresh_logit_val = roc_curve(y_val, logit_scores_val)
plot_roc_curve(fpr_logit_val, tpr_logit_val)
print('AUC Score {}'.format(roc_auc_score(y_val, logit_scores_val)))


# With using our first try with the logistic regression we got an AUC score of .64, not thats good! let's try tuning the parameters to see if we can improve our score. we will try setting a different regularization factor, let's tighten it by 0.1 and 10. and making max_iteration to 1000. Our validation set score is not that far away from our training score and that's a good thing!

# In[334]:


logit_C_low = LogisticRegression(random_state=42, solver="saga", penalty="l1", class_weight="balanced", C=0.001, max_iter=1000)
logit_C_low.fit(X_trained_scaled, y_train)
logit_C_low_scores_proba = logit_C_low.predict_proba(X_trained_scaled)
logit_C_low_scores = logit_C_low_scores_proba[:,1]
fpr_logit_C_low, tpr_logit_C_low, thresh_logit_C_low = roc_curve(y_train, logit_C_low_scores)
#plot_roc_curve(fpr_logit_C_low,tpr_logit_C_low)
print("AUC Score {}".format(roc_auc_score(y_train,logit_C_low_scores)))


# In[337]:


logit_C_high = LogisticRegression(random_state=42, solver="saga", penalty="l1", class_weight="balanced", C=1000, max_iter=1000)
logit_C_high.fit(X_trained_scaled, y_train)
logit_C_high_scores_proba = logit_C_high.predict_proba(X_trained_scaled)
logit_C_high_scores = logit_C_high_scores_proba[:,1]
fpr_logit_C_high, tpr_logit_C_high, thresh_logit_C_high = roc_curve(y_train, logit_C_high_scores)
print("AUC Score {}".format(roc_auc_score(y_train,logit_C_high_scores)))


# In[338]:


#lets make a roc_curve visualization
plt.figure(figsize=(12,10))
plt.plot(fpr_logit, tpr_logit, label="Logit C=1")
plt.plot(fpr_logit_C_high, tpr_logit_C_high , label="Logit C=1000")
plt.plot(fpr_logit_C_low, tpr_logit_C_low , label="Logit C=0.001")
plt.plot([0,1],[0,1], "k--", label="naive prediction")
plt.axis([0,1,0,1])
plt.legend(loc="best")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive rate")


# Adjusting the C parameter don't mean much for our classifier to improve it's score. Let's try our second option which is to implement undersampling of our dataset to make the target variable balanced.

# In[340]:


# Random Sampling
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
print("Original dataset shape {}".format(Counter(y)))


# In[341]:


rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_sample(X,y)
print('Resampled dataset shape {}'.format(Counter(y_resampled)))


# From here, we dropped most of the majority class ended up on a 50/50 ratio. the disadvantage of this strategy is that have lost most of the information from the majority class. advantage are our dataset will have a faster training and we solved the unbalanced dataset problem. let's give it a try!

# In[342]:


X_resampled.shape, y_resampled.shape


# In[343]:


from sklearn.model_selection import train_test_split
X_train_rus, X_val_rus, y_train_rus, y_val_rus = train_test_split(X_resampled, y_resampled,random_state=42)
X_train_rus.shape, y_train_rus.shape


# In[344]:


scaler = StandardScaler().fit(X_train_rus)
X_trained_scaled = scaler.transform(X_train_rus)
X_val_rus_scaled = scaler.transform(X_val_rus)


# In[346]:


logit_resampled = LogisticRegression(random_state=42, solver='saga',
                                     penalty='l1', C=1.0, max_iter=500)
logit_resampled.fit(X_trained_scaled, y_train_rus)
logit_resampled_proba_res = logit_resampled.predict_proba(X_trained_scaled)
logit_resampled_scores = logit_resampled_proba_res[:,1]
fpr_logit_resampled, tpr_logit_resampled, thresh_logit_resampled = roc_curve(y_train_rus, logit_resampled_scores)
plot_roc_curve(fpr_logit_resampled, tpr_logit_resampled)
print("AUC Score {}".format(roc_auc_score(y_train_rus, logit_resampled_scores)))


# Our score doesn't improve that much using the undersampling method. One reason of this would be that the logisticregression model can't handle this vast amount of data or we have reached its limitation of predictive power on this type of dataset. Let's try other complex models!

# One way to improve our score is to use ensembling models. First, we will use RandomForests and will try GradientBoostingClassifier and compare their scores

# In[347]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
forest = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=5,class_weight='balanced')
forest.fit(X_train, y_train) # original dataset
y_scores_proba = forest.predict_proba(X_train)
y_scores = y_scores_proba[:,1]
fpr, tpr, thresh = roc_curve(y_train, y_scores)
plot_roc_curve(fpr,tpr)
print("AUC Score {}".format(roc_auc_score(y_train,y_scores)))


# In[348]:


#Let's cross validate
y_val_proba = forest.predict_proba(X_val)
y_scores_val = y_val_proba[:,1]
fpr_val, tpr_val, thresh_val = roc_curve(y_val, y_scores_val)
plot_roc_curve(fpr_val,tpr_val)
print("AUC Score {}".format(roc_auc_score(y_val,y_scores_val)))


# Let's see how the random forest classifier treat each of the features, and give importance to it.

# In[351]:


def plot_feature_importances(model):
    plt.figure(figsize=(10,8))
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(forest)


# Now GradientBoosting Classifier

# In[353]:


gbc_clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
gbc_clf.fit(X_train,y_train)
gbc_clf_proba = gbc_clf.predict_proba(X_train)
gbc_clf_scores = gbc_clf_proba[:,1]
fpr_gbc, tpr_gbc, thresh_gbc = roc_curve(y_train, gbc_clf_scores)
plot_roc_curve(fpr_gbc, tpr_gbc)
print("AUC Score {}".format(roc_auc_score(y_train, gbc_clf_scores)))


# In[354]:


#validation
gbc_val_proba = gbc_clf.predict_proba(X_val)
gbc_val_scores = gbc_val_proba[:,1]
print("AUC Score {}".format(roc_auc_score(y_val, gbc_val_scores)))


# We are overfitting! Let's try tuning the hyperparameters of our gradient boosting classifier to improve generalization.

# In[355]:


gbc_clf_submission = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05 ,max_depth=4,  random_state=42)
gbc_clf_submission.fit(X_train,y_train)
gbc_clf_proba = gbc_clf_submission.predict_proba(X_train)
gbc_clf_scores = gbc_clf_proba[:,1]
gbc_val_proba = gbc_clf_submission.predict_proba(X_val)
gbc_val_scores = gbc_val_proba[:,1]
fpr_gbc, tpr_gbc, thresh_gbc = roc_curve(y_train, gbc_clf_scores)
print("AUC Score {}".format(roc_auc_score(y_train, gbc_clf_scores))), print("AUC Score {}".format(roc_auc_score(y_val, gbc_val_scores)))


# In[356]:


plot_feature_importances(gbc_clf)


# Here we can see that different alogrithms give importance to different
# features.

# In[360]:




