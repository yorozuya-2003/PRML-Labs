#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import random
import math
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# In[3]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[4]:


import plotly.express as px


# In[5]:


def print_num_unique_entries(df):
    for each in df.columns:
        print('Number of unique entries in', each, ':', len(df[each].unique()))
    
def print_unique_entries(df):
    for each in df.columns:
        print('Unique entries in', each, ':', df[each].unique())


# ---
# # Question-1
# ---

# ## Data preprocessing and Visualization

# In[6]:


df = pd.read_csv('train.csv')
df_raw = df.copy()


# In[7]:


df


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df = df.drop(['Unnamed: 0', 'id'], axis=1)
df


# In[11]:


df.dtypes


# In[12]:


df = df.dropna()


# In[13]:


categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']


# In[14]:


for each in categorical_features:
    df[each] = df[each].astype('category')
    df[each] = df[each].cat.codes


# In[15]:


df


# In[16]:


fig = plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[17]:


# Since Arrival and Departure delays are closely correlated, we could have decided to combine them


# In[18]:


# df['Total Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
# df = df.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
# df


# In[19]:


df.hist(figsize=(20,20))
plt.show()


# In[20]:


df.columns


# In[21]:


features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']


# In[22]:


X = df[features]
y = df['satisfaction']


# In[23]:


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X))
X_scaled.columns = features


# In[24]:


X_scaled


# In[25]:


X = X_scaled


# ### SFS

# In[26]:


sfs1 = SFS(DecisionTreeClassifier(), k_features=10, forward=True, floating=False, scoring='accuracy')


# In[27]:


sfs1 = sfs1.fit(X, y)


# In[28]:


sfs1.subsets_


# In[29]:


sfs1.k_feature_names_


# In[30]:


sfs1.k_score_


# ## Toggling between SFS, SBS, SFFS, and SBFS

# ### Sequential Forward Selection

# In[31]:


sfs = SFS(DecisionTreeClassifier(), k_features=10, forward=True, floating=False, cv=4)
sfs = sfs.fit(X, y)


# In[32]:


sfs.k_score_


# ### Sequential Backward Selection

# In[33]:


sbs = SFS(DecisionTreeClassifier(), k_features=10, forward=False, floating=False, cv=4)
sbs = sbs.fit(X, y)


# In[34]:


sbs.k_score_


# ### Sequential Forward Floating Selection

# In[35]:


sffs = SFS(DecisionTreeClassifier(), k_features=10, forward=True, floating=True, cv=4)
sffs = sffs.fit(X, y)


# In[36]:


sffs.k_score_


# ### Sequential Backward Floating Selection

# In[37]:


sbfs = SFS(DecisionTreeClassifier(), k_features=10, forward=True, floating=False, cv=4)
sbfs = sbfs.fit(X, y)


# In[38]:


sbfs.k_score_


# ## Visualization of the output from the feature selection in Pandas DataFrames

# In[39]:


sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sfs_df


# In[40]:


sbs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sbs_df


# In[41]:


sffs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sffs_df


# In[42]:


sbfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sbfs_df


# ## Plotting the results

# In[43]:


fig = plot_sfs(sfs.get_metric_dict())
plt.title('SFS')
plt.show()


# In[44]:


fig = plot_sfs(sbs.get_metric_dict())
plt.title('SBS')
plt.show()


# In[45]:


fig = plot_sfs(sffs.get_metric_dict())
plt.title('SFFS')
plt.show()


# In[46]:


fig = plot_sfs(sbfs.get_metric_dict())
plt.title('SBFS')
plt.show()


# ## Bi-directional Feature Set Generation Algorithm implementation from scratch

# In[51]:


class BidirectionalFeatureSetGeneration:
    def __init__(self, similarity_measure='accuracy', model='Decision Tree'):
        '''
        Input: similarity measure (default='accuracy')
               estimator model (defaul='Decision Tree')
        
        Constructor for the bidirectional feature set generation class
        '''
        self.similarity_measure = similarity_measure
        self.model = model

        
    def fit(self, X, y, features=None, print_things=True):
        '''
        Input: dataset and labels,
               array of features (if the dataset is not a pandas dataframe),
               an optional parameter to print the iteration information of the feature selection methods
        
        Finds the best features for the input dataset using bidirectional feature set generation
        '''
        self.X = X
        self.y = y
        
        if features == None:
            self.features = X.columns.tolist()
        else:
            self.features = features
            
        train_X, test_X, train_y, test_y = train_test_split(self.X, self.y, random_state=42)
            
        self.distance_measures = {'angular separation': self.angular_separation,
                                  'euclidean distance': self.euclidean_distance,
                                  'city-block distance': self.city_block_distance}
            
        if self.similarity_measure == 'accuracy':
            self.all_features_measure = self.accuracy_measure(train_X, train_y, test_X, test_y)
            self.f_prev_measure = 0
            self.b_prev_measure = 0
        elif self.similarity_measure == 'information gain':
            self.all_features_measure = self.information_gain(train_X, train_y, test_X, test_y)
            self.f_prev_measure = 0
            self.b_prev_measure = 0
        else:
            pred_y = self.get_pred_y(train_X, train_y, test_X)
            self.all_features_measure = self.distance_measures[self.similarity_measure](test_y, pred_y)    
            self.f_prev_measure = float('inf')
            self.b_prev_measure = float('inf')
        
        self.algo(print_things=print_things)
        
    
    def get_pred_y(self, train_X, train_y, test_X, model=None):
        '''
        Input: training dataset and labels,
               testing dataset,
               model for prediction (default = Decision Tree Classifier)
        
        Returns the predictions for the input testing data
        '''
        if model == None:
            model = DecisionTreeClassifier()
            
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        
        return pred_y

    
    def tp_tn_fp_fn(self, train_X, train_y, test_X, test_y, model=None):
        '''
        Input: training dataset and labels,
               testing dataset and labels,
               model for prediction (default = Decision Tree Classifier)
        
        Returns tuple containing (true positives, true neagtives, false positives, false negatives)
        '''
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        test_y = np.array(test_y)
        pred_y = self.get_pred_y(train_X, train_y, test_X, model)
        
        for each in range(len(test_y)):
            if test_y[each]==1 and pred_y[each]==1:
                tp += 1
            elif test_y[each]==1 and pred_y[each]==0:
                fn += 1
            elif test_y[each]==0 and pred_y[each]==1:
                fp += 1
            else:
                tn += 1
        
        return (tp+1e-15, tn+1e-15, fp+1e-15, fn+1e-15)
        
        
    def angular_separation(self, x, y):
        '''
        Input: two datapoints
        
        Returns the euclidean distance between the two datapoints
        '''
        x = np.array(x)
        y = np.array(y)
        return np.sum(x*y) / np.sqrt(np.sum(x**2) * (np.sum(y**2)))

    
    def euclidean_distance(self, x, y):
        '''
        Input: two datapoints
        
        Returns the euclidean distance between the two datapoints
        '''
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(np.array(x) - np.array(y))
    
    
    def city_block_distance(self, x, y):
        '''
        Input: two datapoints
        
        Returns the city-block distance between the inputs
        '''
        x = np.array(x)
        y = np.array(y)
        return np.sum(np.abs(x - y))
    
        
    def accuracy_measure(self, train_X, train_y, test_X, test_y):
        '''
        Input: training dataset and labels,
               testing dataset and labels,
               classifier model to be used for prediction (default = Decision Tree Classifier)
        
        Returns the accuracy score for the input
        '''
        if self.model == 'Decision Tree':
            clf = DecisionTreeClassifier()

        elif self.model == 'SVM':
            clf = SVC()
        
        clf.fit(train_X, train_y)
        return clf.score(test_X, test_y)
        
    
    def information_gain(self, train_X, train_y, test_X, test_y, model=None):
        '''
        Input: training dataset and labels,
               testing dataset and labels,
               classifier model to be used for prediction (default = Decision Tree Classifier)
        
        Returns the information gain based on the input
        '''
        tp, tn, fp, fn = self.tp_tn_fp_fn(train_X, train_y, test_X, test_y, model)
        
        Sum = tp+fp+tn+fn
        Sum_pos = tp+fp
        Sum_neg = tn+fn
        
        return self.e(tp+fn, fp+tn) - ((Sum_pos*self.e(tp,fp) + Sum_neg*self.e(tn,fn)) / Sum)
        
        
    def e(self, a, b):
        '''
        Returns the entropy for the inputs provided
        '''
        Sum = a+b+1e-15
        return -((a/Sum)*math.log2(a/Sum) + (b/Sum)*math.log2(b/Sum))
        
        
    def algo(self, print_things=True):
        '''
        Feature set generation algorithm
        '''
        self.Sf = list()
        self.Sb = self.features.copy()

        f_check_var = 0
        b_check_var = 0
        
        while True:
            # check break condition
            if len(self.Sb)==0:
                break
            
            # forward selection
            ff = self.find_next(print_things=print_things)
            if ff==None:
                f_check_var += 1
                # break
            else:
                self.Sf.append(ff)
                self.Sb.remove(ff)
            
            # check break condition
            if len(self.Sb)==0:
                break
                
            # backward selection
            fb = self.get_next(print_things=print_things)
            if fb==None:
                b_check_var += 1
                # break
            else:
                self.Sb.remove(fb)
            
            if print_things:
                print(f'selected features: {self.Sf}')
                print('*'*100)
                
            if f_check_var == 1 and b_check_var == 0:
                break
            
            elif f_check_var == 0 and b_check_var == 1:
                self.Sf = self.Sf + self.Sb
                break
                
            elif f_check_var == 1 and b_check_var == 1:
                if self.similarity_measure == 'accuracy' or self.similarity_measure == 'information gain':
                    if self.f_prev_measure < self.b_prev_measure:
                        self.Sf = self.Sf + self.Sb
                else:
                    if self.f_prev_measure > self.b_prev_measure:
                        self.Sf = self.Sf + self.Sb
                break
                    
            
        result = self.Sf
        print('- '*50)
        print(f'final feature set:-\n{result}')
        print('- '*50)
        print(f'no. of features: {len(result)}')
        print('- '*50)

        
    def find_next(self, print_things=True):
        '''
        Returns next best feature
        '''
        best_measure = self.f_prev_measure
        best_feature = None
        
        for f in self.Sb:
            X = self.X.copy()
            
            Sf = self.Sf.copy()
            Sf.append(f)
            
            X = X[Sf]
            trainX, testX, trainy, testy = train_test_split(X, self.y, random_state=42)
            
            if self.similarity_measure == 'accuracy':                    
                acc = self.accuracy_measure(trainX, trainy, testX, testy)
                if acc >= best_measure:
                    best_measure = acc
                    best_feature = f
                    
            elif self.similarity_measure == 'information gain':
                info_gain = self.information_gain(trainX, trainy, testX, testy)
                if info_gain >= best_measure:
                    best_measure = info_gain
                    best_feature = f
                    
            else:
                predy = self.get_pred_y(trainX, trainy, testX)
                dist_value = self.distance_measures[self.similarity_measure](testy, predy)
                if dist_value <= best_measure:
                    best_measure = dist_value
                    best_feature = f
        
        if print_things:
            print(f'similarity measure value = {best_measure}, selected feature = {best_feature}')
            
        self.f_prev_measure = best_measure
        return best_feature
    
    
    def get_next(self, print_things=True):
        '''
        Returns next worst feature
        '''
        best_measure = self.b_prev_measure
        worst_feature = None
        
        for f in self.Sb:
            X = self.X.copy()
            
            Sb = self.Sb.copy()
            Sb = Sb + self.Sf.copy()
            Sb.remove(f)
            
            X = X[Sb]
            trainX, testX, trainy, testy = train_test_split(X, self.y, random_state=42)

            if self.similarity_measure == 'accuracy':                    
                acc = self.accuracy_measure(trainX, trainy, testX, testy)
                if acc >= best_measure:
                    best_measure = acc
                    worst_feature = f
                    
            elif self.similarity_measure == 'information gain':
                info_gain = self.information_gain(trainX, trainy, testX, testy)
                if info_gain >= best_measure:
                    best_measure = info_gain
                    worst_feature = f
                    
            else:
                predy = self.get_pred_y(trainX, trainy, testX)
                dist_value = self.distance_measures[self.similarity_measure](testy, predy)
                if dist_value <= best_measure:
                    best_measure = dist_value
                    worst_feature = f
        
        if print_things:
            print(f'similarity measure value = {best_measure}, removed feature = {worst_feature}')
        
        self.b_prev_measure = best_measure
        return worst_feature
    
    
    def result(self):
        '''
        Returns the array of the best features
        '''
        return self.Sf
    
    
    def classification_results(self, model):
        '''
        Input: classifier model of user's choice
        
        Returns the classification results for the 
        selected features set of the data using the classifier
        '''
        train_X, test_X, train_y, test_y = train_test_split(self.X[self.Sf], self.y, random_state=42)
        model.fit(train_X, train_y)
        score = model.score(test_X, test_y)
        print(f'accuracy score: {score}')
    


# ## Showing classification results for the generated feature set 

# In[52]:


def bidirectional_feature_set_generation_different_measures(X, y, measure, model=DecisionTreeClassifier):
    bfsg = BidirectionalFeatureSetGeneration(similarity_measure=measure)
    bfsg.fit(X, y, print_things=False)
    model = DecisionTreeClassifier()
    bfsg.classification_results(model)


# In[53]:


similarity_measures = ['accuracy', 'information gain', 'angular separation',
                       'euclidean distance', 'city-block distance']


# In[54]:


for each in similarity_measures:
    print(f'similarity measure: {each}')
    bidirectional_feature_set_generation_different_measures(X, y, each)
    print()
    print('#'*100)
    print()


# ---
# # Question - 2
# ---

# ## Dataset creation

# In[6]:


cov_mat = np.array([[0.6006771, 0.14889879, 0.244939],
           [0.14889879, 0.58982531, 0.24154981],
           [0.244939, 0.24154981, 0.48778655]])


# In[7]:


cov_mat


# In[8]:


x = np.random.multivariate_normal(mean=[0,0,0], cov=cov_mat, size=1000)


# In[9]:


sqrtsix = math.sqrt(6)
v = np.array([1/sqrtsix, 1/sqrtsix, -2/sqrtsix])


# In[10]:


dotprod = np.dot(x, v)
def label_class(datapoint):
    if datapoint > 0:
        return 0
    return 1
    
classes = np.array([label_class(datapoint) for datapoint in dotprod])
# classes


# ## Data visualization as a 3D scatter-plot

# In[11]:


df = pd.DataFrame(x)
df.columns = ['x', 'y', 'z']
df['label'] = classes
df


# In[12]:


fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')
fig.show()


# ## Principal Component Analysis (n_components=3) on the input dataset

# In[13]:


pca = PCA(n_components=3)
x_transformed = pd.DataFrame(pca.fit_transform(x))
features = [f'feature-{num+1}' for num in range(3)]
x_transformed.columns = features
x_transformed


# ## Complete FS on the transformed data

# In[14]:


def plot_decision_boundary(classifier, train_X, train_y, feature1, feature2, figsize=(7,7)):
    h = 0.02
    
    xf1 = np.array(train_X[feature1])
    xf2 = np.array(train_X[feature2])
    train_y = np.array(train_y)
    
    x_min, x_max = xf1.min() - 10*h, xf1.max() + 10*h
    y_min, y_max = xf2.min() - 10*h, xf2.max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    
    plt.scatter(xf1, xf2, c=train_y, edgecolors='k')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()
    


# In[15]:


def accuracy_and_decision_boundary(model, X, y, features):
    clf = model
    feature1 = features[0]
    feature2 = features[1]
    X_train = X[[feature1, feature2]].copy()
    clf.fit(X_train, y)

    train_X, test_X, train_y, test_y = train_test_split(X_train, y, random_state=0)
    clf_new = model.fit(train_X, train_y)
    acc_score = clf_new.score(test_X, test_y)
    print(f'accuracy score for the features - {feature1} & {feature2}: {acc_score}')
    print('(after train-test split)')
    plot_decision_boundary(clf, X_train, y, feature1, feature2)


# In[16]:


def complete_fs(model, X, y, feature_names, num_features=2):
    feature_combinations = [list(each) for each in combinations(feature_names, num_features)]
    
    for feature_subset in feature_combinations:
        accuracy_and_decision_boundary(model, X, y, feature_subset)
        
        print('-'*100)
        print()
        


# In[17]:


def experimental_comparision(model, X, y, feature_names, num_features=2):
    feature_combinations = [list(each) for each in combinations(feature_names, num_features)]
    
    for feature_subset in feature_combinations:
        pass


# ## Plotting decision boundaries superimposed with the data

# In[18]:


complete_fs(DecisionTreeClassifier(), x_transformed, df['label'], features)


# ## Principal Component Analysis (n_components=2) on the dataset

# In[19]:


pca = PCA(n_components=2)
x_transformed_new = pd.DataFrame(pca.fit_transform(x))
features = [f'feature-{num+1}' for num in range(2)]
x_transformed_new.columns = features
x_transformed_new


# In[20]:


accuracy_and_decision_boundary(DecisionTreeClassifier(), x_transformed_new, df['label'], ['feature-1', 'feature-2'])


# #### The features 'feature-1' and 'feature-2' in the PCA(n_components=3) represent the features obtained by applying PCA(n_components=2) and they also show almost same accuracy scores. These scores are very low as compared to the other two subsets.

# ## Experiments to show the difference in the accuracies

# In[28]:


def plot_learning_curve(X, y, Title):
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)
    model = DecisionTreeClassifier()
    train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=train_X, y=train_y,
                                                    cv=10, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='testing accuracy')
    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')
    plt.title(f'learning curve for {Title}')
    plt.xlabel('train data size')
    plt.ylabel('accuracy score')
    plt.legend()
    plt.show()


# ### learning curves for PCA(n_components=2)

# In[29]:


feature_sets = [['feature-1', 'feature-2'], ['feature-2', 'feature-3'], ['feature-3', 'feature-1']]
feature_set_names = ['feature-1 and feature-2', 'feature-2 and feature-3', 'feature-3 and feature-1']

for each in range(len(feature_sets)):
    plot_learning_curve(x_transformed[feature_sets[each]], df['label'],
                        feature_set_names[each] + str(' PCA(n_components=3)'))


# ### learning curves for PCA(n_components=3)

# In[30]:


plot_learning_curve(x_transformed_new, df['label'], 'PCA(n_components=2)')


# In[ ]:




