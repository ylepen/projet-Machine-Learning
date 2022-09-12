# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:40:41 2022

@author: ylepen
"""

import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt

#%%


Ldata=pd.read_csv("C:/Users/ylepen/Documents/executive_master_big_data/projet ML/project-6-files/learn_dataset.csv",sep=',')

print(list(Ldata.columns))
#%%


Y=Ldata[["target"]]
X=Ldata.iloc[:,1:9]

print(X.shape)
print(list(X.columns))

print(Y.shape)
print(list(Y.columns))

Y.dtypes

X.dtypes

#%%

# descriptive statistics for numerical variables
print(Y.describe())
print(X.describe())

Ldata.head()


#%%
sb.pairplot(Ldata,x_vars=['age_2020','Is_student','DEGREE','act'], y_vars=['target'],hue='Sex')

#%%
Ldata.corr()

sb.boxplot(y='target',x='Sex',data=Ldata)
sb.boxplot(y='target',x='Is_student',data=Ldata)
sb.boxplot(y='target',x='age_2020',data=Ldata)
sb.boxplot(y='target',x='act',data=Ldata)
sb.boxplot(y='target',x='DEGREE',data=Ldata)
sb.boxplot(y='target',x='FAMILTY_TYPE',data=Ldata)
#%%
Xd=pd.get_dummies(X,drop_first=True)

#%%

from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(Xd,Y,
                                        test_size=0.3,
                                        random_state=0)

#%% regression lineaire

from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(train_X.iloc[1:10000,1:5000],train_Y.iloc[1:10000])

#%%

print(reg.intercept_)

#%% forecast

target_f=reg.predict(test_X.iloc[:,1:5000])

#%%
print(target_f)
print(type(target_f))

np.mean(target_f)
np.median(target_f)
np.max(target_f)
np.min(target_f)
#%%

plt.scatter(target_f,test_Y)
