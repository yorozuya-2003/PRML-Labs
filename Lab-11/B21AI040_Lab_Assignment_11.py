#!/usr/bin/env python
# coding: utf-8

# ## Loading libraries and modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay


# ## Loading the dataset

# In[3]:


df = pd.read_csv('data_banknote_authentication.csv')


# In[4]:


df


# ## Performing exploratory analysis & data visualization

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


classes = np.unique(df['class'])
classes


# In[8]:


df.hist(figsize=(10,10))
plt.show()


# In[9]:


fig = plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[10]:


decision_boundary_features = ['variance of Wavelet Transformed image','skewness of Wavelet Transformed image']


# ## Dataset preprocessing

# In[11]:


features = df.columns.tolist()[:-1]
features


# In[12]:


X = df[features].copy()
y = df['class'].copy()


# In[13]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
X.columns = features


# In[14]:


X


# In[15]:


y


# ## Performing train-test-validation split

# In[16]:


train_X, val_test_X, train_y, val_test_y = train_test_split(X, y, test_size=0.3, random_state=0)
test_X, val_X, test_y, val_y = train_test_split(val_test_X, val_test_y, test_size=0.33, random_state=0)


# ## Training models and plotting decision boundaries

# In[17]:


def svm_classifier(X, y, train_X, train_y, test_X, test_y, decision_boundary_features,
                       C=1.0, kernel='rbf', plot=False):
    model = SVC(C=C, kernel=kernel)
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    print(f'test accuracy of SVM (C={C}, kernel={kernel}): {score}')
    
    if plot:
        model = SVC(C=C, kernel=kernel)
        model.fit(X[decision_boundary_features], y)
        h = 0.02

        xf1 = X[decision_boundary_features[0]].to_numpy()
        xf2 = X[decision_boundary_features[1]].to_numpy()
        Y = np.array(y)

        x_min, x_max = xf1.min() - 10*h, xf1.max() + 10*h
        y_min, y_max = xf2.min() - 10*h, xf2.max() + 10*h
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = plt.figure(figsize=(5,5))
        
        color_scheme = 'winter'
        
        plt.contourf(xx, yy, Z, cmap=color_scheme, alpha=0.25)
        plt.contour(xx, yy, Z, cmap=color_scheme, linewidths=0.7)

        plt.scatter(xf1, xf2, c=Y, cmap=color_scheme)
        plt.xlabel(decision_boundary_features[0])
        plt.ylabel(decision_boundary_features[1])
        plt.title(f'decision boundary for SVM (C={C}, kernel={kernel})')
        plt.show()


# In[18]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
Cs = [0.01, 0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100]


# In[19]:


for kernel in kernels:
    for C in Cs:
        svm_classifier(X, y, train_X, train_y, test_X, test_y, decision_boundary_features, C=C, kernel=kernel)
    print()


# In[20]:


for C in Cs:
    svm_classifier(X, y, train_X, train_y, test_X, test_y, decision_boundary_features,
                   C=C, kernel='linear', plot=True)


# In[21]:


for C in Cs:
    svm_classifier(X, y, train_X, train_y, test_X, test_y, decision_boundary_features,
                   C=C, kernel='poly', plot=True)


# In[22]:


for C in Cs:
    svm_classifier(X, y, train_X, train_y, test_X, test_y, decision_boundary_features,
                   C=C, kernel='rbf', plot=True)


# In[23]:


for C in Cs:
    svm_classifier(X, y, train_X, train_y, test_X, test_y, decision_boundary_features,
                   C=C, kernel='sigmoid', plot=True)

