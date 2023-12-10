#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve, mean_squared_error as mse
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score, KFold


# ---
# # Q1
# ---

# ### (1) Perform pre-processing and visualization of the dataset. Split the data into train and test sets. Also identify the useful columns and drop the unnecessary ones.

# In[3]:


df = pd.read_csv('titanic.csv')
df


# In[4]:


df.dtypes


# In[5]:


df.describe()


# In[6]:


df.columns.to_list()


# In[7]:


df.isnull().sum()


# In[8]:


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[9]:


df = df.dropna(axis=0)


# In[10]:


df


# In[11]:


some_features = ['Pclass', 'Age', 'Fare']


# In[12]:


# '''
for feature in some_features:
    df[f'{feature}'].plot.hist()
    plt.xlabel(f'{feature}')
    plt.ylabel('frequency')
    plt.show()
# '''


# In[13]:


df.Sex.describe()


# In[14]:


df.Fare.describe()


# In[15]:


def plotGraph(colName):
    Dict = df[colName].value_counts().to_dict()
    ClassKeys = list(map(str, list(Dict.keys())))
    ClassValues = list(Dict.values())
    plt.bar(ClassKeys, ClassValues)
    plt.xlabel(str(colName))
    plt.ylabel('Number of Passengers')
    plt.show()


# In[16]:


plotGraph('Pclass')


# In[17]:


plotGraph('Sex')


# In[18]:


plotGraph('Embarked')


# In[19]:


df['Pclass'] = df['Pclass'].astype('category')


# In[20]:


classesToBeEncoded = ['Embarked', 'Sex']
for c in classesToBeEncoded:
    df[c] = df[c].astype('category')
    df[c] = df[c].cat.codes


# In[21]:


df


# In[22]:


df.columns.to_list()


# In[23]:


features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']


# In[24]:


continuous_features = ['Age', 'Fare']
for feature in continuous_features:
    sns.distplot(df[feature])
    plt.show()


# In[25]:


X = df[features]
y = df.Survived


# In[26]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)


# ### (3) Implement the identified variant of Naive Bayes Classifier using scikit learn, report its performance based on appropriate metrics (ROC AUC etc.).

# In[27]:


clf = GaussianNB()


# In[28]:


clf.fit(train_X, train_y)


# In[29]:


df


# In[30]:


pred_y = clf.predict(test_X)


# In[31]:


roc_auc_score(test_y, pred_y)


# In[32]:


plot_roc_curve(clf, test_X, test_y)
plt.show()


# ### (4) Perform 5 fold cross validation and summarize the results across the cross-validation sets. Compute the probability of the top class for each row in the testing dataset.

# In[33]:


kf = KFold(n_splits=5)
cv_score = cross_val_score(clf, train_X, train_y, cv=kf)
print(cv_score)
print(cv_score.mean())


# In[34]:


clf.predict_proba(test_X)


# In[35]:


class_probabilities = clf.predict_proba(test_X)


# In[36]:


top_class_probability = []

for each in range(len(class_probabilities)):
    top_class_probability.append(max(class_probabilities[each]))


# In[37]:


top_class_probability


# ### (5) Make contour plots with the data points to visualize the class-conditional densities. 

# In[38]:


for feature in ['Age', 'Fare']:
    plt.figure(figsize=(20,10))
    sns.kdeplot(x=df[feature], y=df['Survived'], cmap="Reds", shade=True)
    plt.scatter(x=df[feature], y=df['Survived'])
    plt.show()


# ### (6) Compare your model with the Decision Tree classifier on the same dataset by performing 5-fold cross-validation and summarizing the results.

# In[39]:


def kfold_cv(train_X, train_y, test_X, test_y, depth=None, split=2, leaf=1):
    dt_clf = DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
    kf = KFold(n_splits=5)
    cv_score = cross_val_score(dt_clf, train_X, train_y, cv=kf)
    print(cv_score)
    print(cv_score.mean())
    
    dt_clf.fit(train_X, train_y)
    plot_roc_curve(dt_clf, test_X, test_y)


# In[40]:


kfold_cv(train_X, train_y, test_X, test_y)


# In[41]:


dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(train_X, train_y)

plot_roc_curve(dt_clf, test_X, test_y)
plot_roc_curve(clf, test_X, test_y)
plt.show()


# ---
# # Q2
# ---

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


df = pd.read_csv('dataset.csv')
df


# **Features**:-  
# X0 = Area  
# X1 = Perimeter  
# X2 = Compactness  
# X3 = Length of kernel  
# X4 = Width of kernel  
# X5 = Asymmetry coefficient  
# X6 = Length of kernel groove  
# X7 =  Class (1, 2, 3)  

# ### (a) Use histogram to plot the distribution of samples.

# In[44]:


df.hist(figsize=(20,20))
plt.show()


# ### (b) Determine the prior probability for all the classes.

# In[45]:


df['Y'].unique()


# Hence, total unique classes are 3.

# In[46]:


def prior(Class):
    return len(df[df['Y']==Class])/len(df)


# In[47]:


prior1 = prior(1)
prior2 = prior(2)
prior3 = prior(3)


# In[48]:


print('prior(ω1) =', prior1, '\nprior(ω2) =', prior2, '\nprior(ω3) =',prior3)


# In[49]:


prior1 + prior2 + prior3


# ### (c) Discretize the features into bins from scratch.

# In[50]:


def binning_using_difference(df, feature_name, diff):
    min_element = df[feature_name].min()
    max_element = df[feature_name].max()
    
    list_of_lists_of_indices = []
    list_of_bin_names = []
    
    num = round(int(min_element/diff)*diff, 2)
    
    while(max_element - num > 0):
        df_needed = df[feature_name]
        df_needed = df_needed[df_needed >= num]
        df_needed = df_needed[df_needed < num + diff]
        list_of_indices = list(df_needed.index)
        
        list_of_lists_of_indices.append(list_of_indices)
        list_of_bin_names.append(f'{num:.2f}-{(num+diff):.2f}')
        
        num += diff

    df[f'bin-{feature_name}'] = 0
        
    for l in range(len(list_of_lists_of_indices)):
        corresponding_name = list_of_bin_names[l]
        if len(list_of_lists_of_indices[l]) > 0:
            for ind in list_of_lists_of_indices[l]:
                df[f'bin-{feature_name}'][ind] = corresponding_name
            
    return df


# In[51]:


features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']
diff_for_features = [2, 1, 0.02, 0.25, 0.2, 1, 0.25]


# In[52]:


for i in range(len(features)):
    df = binning_using_difference(df, features[i], diff_for_features[i])


# In[53]:


df


# ### (d) Determine the likelihood/class conditional probabilities for all the classes.

# class conditional = p(x | ωj) = p(x ∩ ωj) / p(ωj)

# In[54]:


new_df = pd.read_csv('dataset.csv')


# In[55]:


def new_binning_using_difference(df, feature_name, diff):
    min_element = df[feature_name].min()
    max_element = df[feature_name].max()
    
    list_of_lists_of_indices = []
    list_of_bin_names = []
    
    num = round(int(min_element/diff)*diff, 2)
    
    name = 1
    while(max_element - num > 0):
        df_needed = df[feature_name]
        df_needed = df_needed[df_needed >= num]
        df_needed = df_needed[df_needed < num + diff]
        list_of_indices = list(df_needed.index)
        
        list_of_lists_of_indices.append(list_of_indices)
        list_of_bin_names.append(name)
        
        num += diff
        name += 1

    df[f'bin-{feature_name}'] = 0
        
    for l in range(len(list_of_lists_of_indices)):
        corresponding_name = list_of_bin_names[l]
        if len(list_of_lists_of_indices[l]) > 0:
            for ind in list_of_lists_of_indices[l]:
                df[f'bin-{feature_name}'][ind] = corresponding_name
            
    return df


# In[56]:


features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']
diff_for_features = [2, 1, 0.02, 0.25, 0.2, 1, 0.25]


# In[57]:


for i in range(len(features)):
    new_df = new_binning_using_difference(new_df, features[i], diff_for_features[i])


# In[58]:


prior_dict = {1: prior1, 2: prior2, 3: prior3}

def likelihood(new_df, Feature, Bin, Class):
    L = len(new_df)
    
    new_df = new_df[new_df[Feature]==Bin]
    new_df = new_df[new_df['Y']==Class]
    
    return len(new_df)/(L * prior_dict[Class])


# In[59]:


for Feature in ['bin-X0', 'bin-X1', 'bin-X2', 'bin-X3', 'bin-X4', 'bin-X5', 'bin-X6']:
    length = len(new_df[Feature].unique())
    
    for each in range(1, length+1):
        for Class in range(1,4):
            print(f'Likelihood for bin no.:{each}, in {Feature[4:7]} for class {Class} =', likelihood(new_df, Feature, each, Class))
        print('\n')
    print('-'*70,'\n')


# ### (e) Plot the count of each unique element for each class.

# In[60]:


for i in range(0,7):
    print(df.groupby(f'bin-X{i}')['Y'].value_counts())
    print('\n')


# In[61]:


for i in range(0,7):
    print(f'bin-X{i} =', dict(df.groupby(f'bin-X{i}')['Y'].value_counts()))
    print('\n')


# In[62]:


new_df


# In[63]:


for i in range(0,7):
    print(f'bin_X{i} =', dict(new_df.groupby(f'bin-X{i}')['Y'].value_counts()))
    print('\n')


# In[64]:


def plotter(d, Title):
    d1 = {}; d2 = {}; d3 = {}
    max_bin = 1
    for key,val in d.items():
        feature, y = key
        if y==1:
            d1[feature] = val
        elif y==2:
            d2[feature] = val
        elif y==3:
            d3[feature] = val
        max_bin = max(max_bin, feature)
        
    x = []
    for i in range(1,max_bin+1):
        x.append(i)
        
    y1 = [0 for i in range(max_bin)]
    for i in d1.keys():
        y1[i-1] = d1[i]
    
    y2 = [0 for i in range(max_bin)]
    for i in d2.keys():
        y2[i-1] = d2[i]
        
    y3 = [0 for i in range(max_bin)]
    for i in d3.keys():
        y3[i-1] = d3[i]
        
    plt.plot(x,y1,'o-y')
    plt.plot(x,y2,'o-m')
    plt.plot(x,y3,'o-c')
    
    plt.xlabel('bins')
    plt.ylabel('count')
    plt.title(f'Count of each unique element for each class for {Title}')
    
    plt.show()


# In[65]:


bin_X0 = {(1, 3): 37, (1, 1): 2, (2, 3): 33, (2, 1): 22, (3, 1): 39, (3, 2): 6, (4, 2): 18, (4, 1): 7, (5, 2): 37, (6, 2): 9}


bin_X1 = {(1, 3): 16, (1, 1): 2, (2, 3): 54, (2, 1): 16, (3, 1): 46, (3, 2): 4, (4, 2): 22, (4, 1): 6, (5, 2): 39, (6, 2): 5}


bin_X2 = {(1, 3): 8, (2, 3): 16, (2, 1): 1, (3, 3): 26, (3, 1): 6, (3, 2): 6, (4, 1): 28, (4, 2): 24, (4, 3): 14, (5, 2): 30, (5, 1): 27, (5, 3): 6, (6, 2): 10, (6, 1): 8}


bin_X3 = {(1, 3): 3, (1, 1): 1, (2, 3): 38, (2, 1): 9, (3, 3): 28, (3, 1): 21, (3, 2): 2, (4, 1): 30, (4, 2): 1, (4, 3): 1, (5, 2): 17, (5, 1): 8, (6, 2): 26, (6, 1): 1, (7, 2): 16, (8, 2): 8}


bin_X4 = {(1, 3): 29, (2, 3): 30, (2, 1): 6, (3, 1): 24, (3, 3): 10, (4, 1): 26, (4, 2): 5, (4, 3): 1, (5, 2): 21, (5, 1): 13, (6, 2): 25, (6, 1): 1, (7, 2): 17, (8, 2): 2}


bin_X5 = {(1, 1): 3, (2, 1): 18, (2, 2): 5, (2, 3): 1, (3, 1): 26, (3, 2): 19, (3, 3): 4, (4, 2): 19, (4, 1): 14, (4, 3): 12, (5, 3): 27, (5, 2): 17, (5, 1): 6, (6, 3): 16, (6, 2): 8, (6, 1): 2, (7, 3): 6, (7, 2): 2, (7, 1): 1, (8, 3): 2, (9, 3): 2}


bin_X6 = {(1, 1): 7, (1, 3): 1, (2, 1): 15, (2, 3): 11, (3, 3): 41, (3, 1): 32, (3, 2): 1, (4, 3): 17, (4, 1): 12, (4, 2): 1, (5, 2): 4, (5, 1): 3, (6, 2): 30, (6, 1): 1, (7, 2): 22, (8, 2): 11, (9, 2): 1}


# In[66]:


all_bins_dict = [bin_X0, bin_X1, bin_X2, bin_X3, bin_X4, bin_X5, bin_X6]


# In[67]:


for b in range(len(all_bins_dict)):
    plotter(all_bins_dict[b], Title=features[b])


# ### (f) Calculate the posterior probabilities and plot them in a single graph.

# In[68]:


def posterior(new_df, Feature, Bin, Class):
    L = len(new_df[new_df[Feature]==Bin])
    
    new_df = new_df[new_df[Feature]==Bin]
    new_df = new_df[new_df['Y']==Class]
    
    return len(new_df)/L


# In[69]:


def plot_posterior(new_df, Feature, Title):
    length = len(new_df[Feature].unique())
    
    x = []
    for i in range(1,length+1):
        x.append(i)

    y1 = [0 for i in range(length)]
    for i in range(length):
        y1[i] = posterior(new_df, Feature, i+1, 1)

    y2 = [0 for i in range(length)]
    for i in range(length):
        y2[i] = posterior(new_df, Feature, i+1, 2)

    y3 = [0 for i in range(length)]
    for i in range(length):
        y3[i] = posterior(new_df, Feature, i+1, 3)

    plt.plot(x,y1,'o-y')
    plt.plot(x,y2,'o-m')
    plt.plot(x,y3,'o-c')

    plt.xlabel('bins')
    plt.ylabel('posterior probabilities')
    plt.title(f'posterior probabilities for {Title}')

    plt.show()


# In[70]:


for feature in ['bin-X0', 'bin-X1', 'bin-X2', 'bin-X3', 'bin-X4', 'bin-X5', 'bin-X6']:
    plot_posterior(new_df, feature, feature[4:7])


# In[ ]:




