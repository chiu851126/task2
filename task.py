import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

#讀取資料
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

file_info = train.info()
file_info_test = test.info()
#資料欄位統計
file_describe = train.describe()
file_describe_test = test.describe()

a = train.isnull().sum()
a1 = test.isnull().sum()

#%% train
#資料欄位類別型態(畫長條圖)
#地理位置
sns.barplot(x = train['Geography'].value_counts().index, y = train['Geography'].value_counts())
plt.title('Geography')
plt.xlabel('Geography')
plt.ylabel('Counts')
plt.show()

#年齡
sns.barplot(x = train['Gender'].value_counts().index, y = train['Gender'].value_counts())
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Counts')
plt.show()

#CreditScore
sns.histplot(train['CreditScore'], binwidth=10)
plt.title('CreditScore', fontsize=15)
plt.xlabel('CreditScore',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Age
#sns.histplot(train['Age'], binwidth=3)
sns.boxplot(train['Age'])
plt.title('Age', fontsize=15)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Tenure
sns.barplot(x = train['Tenure'].value_counts().index, y = train['Tenure'].value_counts())
plt.title('Tenure')
plt.xlabel('Tenure')
plt.ylabel('Counts')
plt.show()

#Balance
sns.histplot(train['Balance'], bins=15, alpha=0.7)
plt.title('Balance', fontsize=10)
plt.xlabel('Balance', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.show()

#NumOfProducts
sns.barplot(x = train['NumOfProducts'].value_counts().index, y = train['NumOfProducts'].value_counts())
plt.title('NumOfProducts')
plt.xlabel('NumOfProducts')
plt.ylabel('Counts')
plt.show()

#HasCrCard
sns.barplot(x = train['HasCrCard'].value_counts().index, y = train['HasCrCard'].value_counts())
plt.title('HasCrCard')
plt.xlabel('HasCrCard')
plt.ylabel('Counts')
plt.show()

#IsActiveMember
sns.barplot(x = train['IsActiveMember'].value_counts().index, y = train['IsActiveMember'].value_counts())
plt.title('IsActiveMember')
plt.xlabel('IsActiveMember')
plt.ylabel('Counts')
plt.show()

#EstimatedSalary
sns.histplot(train['EstimatedSalary'], bins=15, alpha=0.7)
plt.title('EstimatedSalary')
plt.xlabel('EstimatedSalary')
plt.ylabel('Counts')
plt.show()

#Exited
sns.barplot(x = train['Exited'].value_counts().index, y = train['Exited'].value_counts())
plt.title('Exited')
plt.xlabel('Exited')
plt.ylabel('Counts')
plt.show()

#%% test
#資料欄位類別型態(畫長條圖)
#地理位置
sns.barplot(x = test['Geography'].value_counts().index, y = test['Geography'].value_counts())
plt.title('Geography')
plt.xlabel('Geography')
plt.ylabel('Counts')
plt.show()

#年齡
sns.barplot(x = test['Gender'].value_counts().index, y = test['Gender'].value_counts())
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Counts')
plt.show()

#CreditScore
sns.histplot(test['CreditScore'], binwidth=10)
plt.title('CreditScore', fontsize=15)
plt.xlabel('CreditScore',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Age
#sns.histplot(train['Age'], binwidth=3)
sns.boxplot(test['Age'])
plt.title('Age', fontsize=15)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Tenure
sns.barplot(x = test['Tenure'].value_counts().index, y = test['Tenure'].value_counts())
plt.title('Tenure')
plt.xlabel('Tenure')
plt.ylabel('Counts')
plt.show()

#Balance
sns.histplot(test['Balance'], bins=15, alpha=0.7)
plt.title('Balance', fontsize=10)
plt.xlabel('Balance', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.show()

#NumOfProducts
sns.barplot(x = test['NumOfProducts'].value_counts().index, y = test['NumOfProducts'].value_counts())
plt.title('NumOfProducts')
plt.xlabel('NumOfProducts')
plt.ylabel('Counts')
plt.show()

#HasCrCard
sns.barplot(x = test['HasCrCard'].value_counts().index, y = test['HasCrCard'].value_counts())
plt.title('HasCrCard')
plt.xlabel('HasCrCard')
plt.ylabel('Counts')
plt.show()

#IsActiveMember
sns.barplot(x = test['IsActiveMember'].value_counts().index, y = test['IsActiveMember'].value_counts())
plt.title('IsActiveMember')
plt.xlabel('IsActiveMember')
plt.ylabel('Counts')
plt.show()

#EstimatedSalary
sns.histplot(test['EstimatedSalary'], bins=15, alpha=0.7)
plt.title('EstimatedSalary')
plt.xlabel('EstimatedSalary')
plt.ylabel('Counts')
plt.show()

#%% train資料前處理

#Outlier
for q in train[['Age', 'CreditScore']]:
    Q1 = train[q].quantile(0.25)
    Q3 = train[q].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    train.drop(train[(train[q] >upper)|(train[q] <lower)].index, inplace=True)

    
    print('25% = ',Q1)
    print('75% =',Q3)
    print('IQR =',IQR)
    print('lower =',lower)
    print('upper =',upper)
    print(train) 
    print('......')


#地理位置 類別型態轉數值(nominal)onehotencoding 
onehotencoder = OneHotEncoder()
train['Geography'] = onehotencoder.fit_transform(train[['Geography']]).toarray()

#年齡 類別型態轉數值型態(binary)
train['Gender'] = onehotencoder.fit_transform(train[['Gender']]).toarray()

#建立MinMaxScaler物件
minmax = preprocessing.MinMaxScaler()
# 資料標準化
train['CreditScore'] = minmax.fit_transform(train[['CreditScore']])
train['Age'] = minmax.fit_transform(train[['Age']])
train['Tenure'] = minmax.fit_transform(train[['Tenure']])
train['Balance'] = minmax.fit_transform(train[['Balance']])
train['EstimatedSalary'] = minmax.fit_transform(train[['EstimatedSalary']])

#CreditScore
sns.histplot(train['CreditScore'], bins=15, alpha=0.7)
plt.title('CreditScore', fontsize=15)
plt.xlabel('CreditScore',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Age
sns.histplot(train['Age'], bins=15, alpha=0.7)
plt.title('Age', fontsize=15)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Tenure
sns.histplot(train['Tenure'], bins=15, alpha=0.7)
plt.title('Tenure')
plt.xlabel('Tenure')
plt.ylabel('Counts')
plt.show()

#Balance
sns.histplot(train['Balance'], bins=15, alpha=0.7)
plt.title('Balance', fontsize=10)
plt.xlabel('Balance', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.show()

#EstimatedSalary
sns.histplot(train['EstimatedSalary'], bins=15, alpha=0.7)
plt.title('EstimatedSalary')
plt.xlabel('EstimatedSalary')
plt.ylabel('Counts')
plt.show()

sns.barplot(x = train['Exited'].value_counts().index, y = train['Exited'].value_counts())
plt.title('Exited')
plt.xlabel('Exited')
plt.ylabel('Counts')
plt.show()

#%% test資料前處理

#地理位置 類別型態轉數值(nominal)onehotencoding 
onehotencoder = OneHotEncoder()
test['Geography'] = onehotencoder.fit_transform(test[['Geography']]).toarray()

#年齡 類別型態轉數值型態(binary)
test['Gender'] = onehotencoder.fit_transform(test[['Gender']]).toarray()

#建立MinMaxScaler物件
minmax = preprocessing.MinMaxScaler()
# 資料標準化
test['CreditScore'] = minmax.fit_transform(test[['CreditScore']])
test['Age'] = minmax.fit_transform(test[['Age']])
test['Tenure'] = minmax.fit_transform(test[['Tenure']])
test['Balance'] = minmax.fit_transform(test[['Balance']])
test['EstimatedSalary'] = minmax.fit_transform(test[['EstimatedSalary']])

#CreditScore
sns.histplot(test['CreditScore'], bins=15, alpha=0.7)
plt.title('CreditScore', fontsize=15)
plt.xlabel('CreditScore',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Age
sns.histplot(test['Age'], bins=15, alpha=0.7)
plt.title('Age', fontsize=15)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.show()

#Tenure
sns.histplot(test['Tenure'], bins=15, alpha=0.7)
plt.title('Tenure')
plt.xlabel('Tenure')
plt.ylabel('Counts')
plt.show()

#Balance
sns.histplot(test['Balance'], bins=15, alpha=0.7)
plt.title('Balance', fontsize=10)
plt.xlabel('Balance', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.show()

#EstimatedSalary
sns.histplot(test['EstimatedSalary'], bins=15, alpha=0.7)
plt.title('EstimatedSalary')
plt.xlabel('EstimatedSalary')
plt.ylabel('Counts')
plt.show()

#%%
X = train.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis = 1)
Y = train['Exited']
X1 = test.drop(['id', 'CustomerId', 'Surname'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=123456)

models = []
models.append(('LogisticRegression', LogisticRegression(random_state = 12345)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RandomForest', RandomForestClassifier(random_state = 12345)))
#models.append(('SVM', SVC(kernel= 'linear', gamma='auto', random_state = 12345)))
#models.append(('XGB', XGBClassifier(random_state = 12345)))

alog_1 = []
alog_2 = []
alog_1_neg = []
alog_2_neg = []

for name, model in models:
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    val_accuracy = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy') #10倍交叉驗證
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) #精確率
    recall = recall_score(y_test, y_pred)       #召回率
    F1 = f1_score(y_test, y_pred)               #F1 measure
    conf = confusion_matrix(y_test, y_pred)     #混淆矩陣

    print('%s:' %name)
    print('val_accuracy: %.4f' %val_accuracy.mean())
    print('accuracy: %.4f' %accuracy)
    print('precision: %.4f' %precision)
    print('recall: %.4f' %recall)
    print('F1: %.4f' %F1)
    print("conf:\n", conf)
    pre, re, thresholds = precision_recall_curve(y_test, y_pred)
    
    alog_1.append(pre)
    alog_2.append(re)
#%%
arr_1_1 = np.array(alog_1[0]).T
arr_1_2 = np.array(alog_1[1]).T
arr_1_3 = np.array(alog_1[2]).T
arr_1_4 = np.array(alog_1[3]).T
arr_1_5 = np.array(alog_1[4]).T
arr_2_1 = np.array(alog_2[0]).T
arr_2_2 = np.array(alog_2[1]).T
arr_2_3 = np.array(alog_2[2]).T
arr_2_4 = np.array(alog_2[3]).T
arr_2_5 = np.array(alog_2[4]).T
#%%
 
auc_score = auc(arr_2_1, arr_1_1)
auc_score_1 = auc(arr_2_2, arr_1_2)
auc_score_2 = auc(arr_2_3, arr_1_3)
auc_score_3 = auc(arr_2_4, arr_1_4)
auc_score_4 = auc(arr_2_5, arr_1_5)

plt.plot(arr_2_1, arr_1_1, color='b', label=f'LogisticRegression AUC = {auc_score:.2f}')
plt.plot(arr_2_2, arr_1_2, color='g', label=f'KNN AUC = {auc_score_1:.2f}')
plt.plot(arr_2_3, arr_1_3, color='r', label=f'RandomForest AUC = {auc_score_2:.2f}')
plt.plot(arr_2_4, arr_1_4, color='m', label=f'SVM AUC = {auc_score_3:.2f}')
plt.plot(arr_2_5, arr_1_5, color='c', label=f'XGB AUC = {auc_score_4:.2f}')

plt.xlabel('Recall (Class=1)')
plt.ylabel('Precision (Class=1)')
plt.title('Precision-Recall Curve (Class=1)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid(True)
plt.fill_between(arr_2_1, arr_1_1, alpha=0.1, color='b')  # 填充PR曲線下的面積
plt.fill_between(arr_2_2, arr_1_2, alpha=0.1, color='g')  # 填充PR曲線下的面積
plt.fill_between(arr_2_3, arr_1_3, alpha=0.1, color='r')  # 填充PR曲線下的面積
plt.fill_between(arr_2_4, arr_1_4, alpha=0.1, color='m')  # 填充PR曲線下的面積
plt.fill_between(arr_2_5, arr_1_5, alpha=0.1, color='c')  # 填充PR曲線下的面積
plt.legend(loc='lower left')
plt.show()

#%%
'''
#SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train,y_train)

alog_1_sm = []
alog_2_sm = []
alog_1_neg = []
alog_2_neg = []

for name, model_sm in models:
    
    model_sm.fit(X_smote, y_smote)
    y_pred_sm = model_sm.predict(X_test)
    val_accuracy_sm = cross_val_score(model, X_smote, y_smote, cv = 10, scoring = 'accuracy') #10倍交叉驗證
    accuracy_sm = accuracy_score(y_test, y_pred_sm)
    precision_sm = precision_score(y_test, y_pred_sm) #精確率
    recall_sm = recall_score(y_test, y_pred_sm)       #召回率
    F1_sm = f1_score(y_test, y_pred_sm)               #F1 measure
    conf_sm = confusion_matrix(y_test, y_pred_sm)     #混淆矩陣

    print('%s:' %name)
    print('val_accuracy: %.4f' %val_accuracy_sm.mean())
    print('accuracy: %.4f' %accuracy_sm)
    print('precision: %.4f' %precision_sm)
    print('recall: %.4f' %recall_sm)
    print('F1: %.4f' %F1_sm)
    print("conf:\n", conf_sm)
    pre_sm, re_sm, thresholds = precision_recall_curve(y_test, y_pred_sm)
    
    alog_1_sm.append(pre_sm)
    alog_2_sm.append(re_sm)
#%%
arr_1_1_sm = np.array(alog_1_sm[0]).T
arr_1_2_sm = np.array(alog_1_sm[1]).T
arr_1_3_sm = np.array(alog_1_sm[2]).T
arr_1_4_sm = np.array(alog_1_sm[3]).T
arr_1_5_sm = np.array(alog_1_sm[4]).T
arr_2_1_sm = np.array(alog_2_sm[0]).T
arr_2_2_sm = np.array(alog_2_sm[1]).T
arr_2_3_sm = np.array(alog_2_sm[2]).T
arr_2_4_sm = np.array(alog_2_sm[3]).T
arr_2_5_sm = np.array(alog_2_sm[4]).T
#%%
 
auc_score_sm = auc(arr_2_1, arr_1_1)
auc_score_1_sm = auc(arr_2_2, arr_1_2)
auc_score_2_sm= auc(arr_2_3, arr_1_3)
auc_score_3_sm = auc(arr_2_4, arr_1_4)
auc_score_4_sm = auc(arr_2_5, arr_1_5)

plt.plot(arr_2_1_sm, arr_1_1_sm, color='b', label=f'LogisticRegression AUC = {auc_score:.2f}')
plt.plot(arr_2_2_sm, arr_1_2_sm, color='g', label=f'KNN AUC = {auc_score_1:.2f}')
plt.plot(arr_2_3_sm, arr_1_3_sm, color='r', label=f'RandomForest AUC = {auc_score_1:.2f}')
plt.plot(arr_2_4_sm, arr_1_4_sm, color='m', label=f'SVM AUC = {auc_score_1:.2f}')
plt.plot(arr_2_5_sm, arr_1_5_sm, color='c', label=f'XGB AUC = {auc_score_1:.2f}')

plt.xlabel('Recall (Class=1)')
plt.ylabel('Precision (Class=1)')
plt.title('Precision-Recall Curve (Class=1)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid(True)
plt.fill_between(arr_2_1_sm, arr_1_1_sm, alpha=0.1, color='b')  # 填充PR曲線下的面積
plt.fill_between(arr_2_2_sm, arr_1_2_sm, alpha=0.1, color='g')  # 填充PR曲線下的面積

plt.fill_between(arr_2_3_sm, arr_1_3_sm, alpha=0.1, color='r')  # 填充PR曲線下的面積
plt.fill_between(arr_2_4_sm, arr_1_4_sm, alpha=0.1, color='m')  # 填充PR曲線下的面積
plt.fill_between(arr_2_5_sm, arr_1_5_sm, alpha=0.1, color='c')  # 填充PR曲線下的面積
plt.legend(loc='lower left')
plt.show()
'''