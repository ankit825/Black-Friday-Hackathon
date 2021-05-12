# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:55:53 2021

@author: Ankit Solanki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# Loading and understanding data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Checking the variable names and the first 5 rows of the data frame and info

train.head()
train.info()
train.describe(include = 'all')


# TRAIN DATA


train_x = train.drop(columns =['User_ID','Product_ID'],axis=1)

#TO CHECK MISSING VALUES
train_x.isnull().sum()
train_x.isnull().sum()/train_x.shape[0]*100
 
 '''Product_Category_3 is null for nearly 70% of transactions so it can't 
 give us much information. so we gonna drop Product_Category_3''' 

train_x = train_x.drop(columns =['Product_Category_3'],axis=1)

train_x['Product_Category_2'].fillna(0, inplace=True)



train_x= pd.get_dummies(train_x)

# TO CHECK COREELATION
plt.figure(figsize=(14,6))
corr= train_x.corr()
sns.heatmap(corr, linewidths=1.5, annot= True)
plt.show()

train_x = train_x.drop(columns =['Marital_Status'],axis=1)

# SPLITTING OF DATA
X= train_x.drop(["Purchase"], axis=1)
y= train_x["Purchase"]
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)






# TEST DATA


test_x = test.drop(columns =['User_ID','Product_ID'],axis=1)

#TO CHECK MISSING VALUES
test_x.isnull().sum()
test_x.isnull().sum()/test_x.shape[0]*100
 
 '''Product_Category_3 is null for nearly 70% of transactions so it can't 
 give us much information. so we gonna drop Product_Category_3''' 

test_x = test_x.drop(columns =['Product_Category_3'],axis=1)


test_x['Product_Category_2'].fillna(0, inplace=True)



test_x= pd.get_dummies(test_x)

# TO CHECK COREELATION
plt.figure(figsize=(14,6))
corr= test_x.corr()
sns.heatmap(corr, linewidths=1.5, annot= True)
plt.show()

test_x = test_x.drop(columns =['Marital_Status'],axis=1)


dtrain = xgb.DMatrix(train_data, label=train_labels)
dtest = xgb.DMatrix(test_data, label=test_labels)

dpred = xgb.DMatrix(test_x)

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:squarederror',
}


params['eval_metric'] = "rmse"
num_boost_round = 999


model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
    
)

y_pred = model.predict(dpred)

submission = pd.DataFrame(y_pred, columns = ['Purchase'])
submission['User_ID'] = test['User_ID']
submission['Product_ID'] = test['Product_ID']
submission.head()

submission.to_csv('XGboost_submission.csv')



