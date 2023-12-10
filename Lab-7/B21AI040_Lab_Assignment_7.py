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

import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


def print_num_unique_entries(df):
    for each in df.columns:
        print('Number of unique entries in', each, ':', len(df[each].unique()))
    
def print_unique_entries(df):
    for each in df.columns:
        print('Unique entries in', each, ':', df[each].unique())


# ---
# # Question-1
# ---

# In[4]:


anneal_data = pd.read_csv('anneal_data.csv')
anneal_test = pd.read_csv('anneal_test.csv')
anneal_df = anneal_data.append(anneal_test)
anneal_df


# In[5]:


anneal_df.isnull().sum()


# In[6]:


drop_cols = []
null_col_dict = anneal_df.isnull().sum().to_dict()
for each in null_col_dict.keys():
    if null_col_dict[each] > (898//2):
        drop_cols.append(each)
drop_cols


# In[7]:


anneal_df = anneal_df.drop(drop_cols, axis=1)
anneal_df


# ### Data Imputation

# In[8]:


# storing most frequently occuring value in each column
col_modes_dict = anneal_df.mode().to_dict()


# In[9]:


cols_to_modify = []
null_col_dict = anneal_df.isnull().sum().to_dict()
for each in null_col_dict.keys():
    if null_col_dict[each] > 0:
        cols_to_modify.append(each)
cols_to_modify


# In[10]:


for each in cols_to_modify:
    anneal_df[each] = anneal_df[each].replace(np.nan, col_modes_dict[each][0])


# In[11]:


anneal_df


# In[12]:


anneal_df.isnull().sum()


# In[13]:


anneal_df = anneal_df.drop('product-type', axis=1)


# In[14]:


anneal_df


# In[15]:


fig = plt.figure(figsize=(15,10))
dataplot = sns.heatmap(anneal_df.corr(), annot=True)
plt.show()


# In[16]:


print_num_unique_entries(anneal_df)


# In[17]:


print_unique_entries(anneal_df)


# In[18]:


anneal_df.dtypes


# In[19]:


anneal_df['classes'] = anneal_df['classes'].replace('U', '6')


# In[20]:


print_unique_entries(anneal_df)


# In[21]:


anneal_df.hist(figsize=(15,15))
plt.show()


# In[22]:


categorical_features = ['steel', 'condition', 'surface-quality', 'shape', 'classes']


# In[23]:


for each in categorical_features:
    anneal_df[each] = anneal_df[each].astype('category')
    anneal_df[each] = anneal_df[each].cat.codes


# In[24]:


anneal_df


# In[25]:


continuous_features = ['carbon', 'hardness', 'strength', 'thick', 'width', 'len']


# In[26]:


X_std = anneal_df[anneal_df.columns.tolist()[:-1]].copy()
y_std = anneal_df['classes'].copy()

X_non_std = X_std.copy()
y_non_std = y_std.copy()


# In[27]:


scaler = StandardScaler()
X_std = scaler.fit_transform(X_std)
X_std = pd.DataFrame(X_std)
X_std.columns = X_non_std.columns


# In[28]:


X_std


# ### Train-test split

# In[29]:


train_X_std, test_X_std, train_y_std, test_y_std = train_test_split(X_std, y_std, test_size=0.35, random_state=0)
train_X_non_std, test_X_non_std, train_y_non_std, test_y_non_std = train_test_split(X_non_std, y_non_std, test_size=0.35, random_state=0)


# ### On standardised data

# In[30]:


rf_model_1 = RandomForestClassifier()
bagging_model_1 = BaggingClassifier()
svm_clf_1 = SVC()

rf_model_1.fit(train_X_std, train_y_std)
bagging_model_1.fit(train_X_std, train_y_std)
svm_clf_1.fit(train_X_std, train_y_std)


# In[31]:


rf_model_1.score(test_X_std, test_y_std)


# In[32]:


bagging_model_1.score(test_X_std, test_y_std)


# In[33]:


svm_clf_1.score(test_X_std, test_y_std)


# ### On non-standardized data

# In[34]:


rf_model_2 = RandomForestClassifier()
bagging_model_2 = BaggingClassifier()
svm_clf_2 = SVC()

rf_model_2.fit(train_X_non_std, train_y_non_std)
bagging_model_2.fit(train_X_non_std, train_y_non_std)
svm_clf_2.fit(train_X_non_std, train_y_non_std)


# In[35]:


rf_model_2.score(test_X_non_std, test_y_non_std)


# In[36]:


bagging_model_2.score(test_X_non_std, test_y_non_std)


# In[37]:


svm_clf_2.score(test_X_non_std, test_y_non_std)


# ## 5-fold cross-validation scores and plots

# In[38]:


def cross_validation_plots(models, X, y, model_names, title, n_splits=5):
    kf = KFold(n_splits=n_splits)
    x_array = [x+1 for x in range(n_splits)]
    for each in range(len(models)):
        cv_score = np.array(cross_val_score(models[each], X, y, cv=kf))
        print(f'{n_splits}-fold-cross-validation scores for {model_names[each]}: {cv_score}')
        plt.plot(x_array, cv_score, label=model_names[each])
    plt.legend()
    plt.xlabel('fold no.')
    plt.ylabel('cross validation accuracy scores')
    plt.title(title)
    plt.show()


# In[39]:


model_names = ['RandomForestClassifier', 'BaggingClassifier', 'SVMClassifier']
models = [RandomForestClassifier(), BaggingClassifier(), SVC()]


# ### On standardised data

# In[40]:


cross_validation_plots(models, X_std, y_std, model_names, title='for standardized data')


# ### On non-standardised data

# In[41]:


cross_validation_plots(models, X_non_std, y_non_std, model_names, title='for non-standardized data')


# ## Principal Component Analysis implementation from scratch

# In[42]:


class PCA:
    def __init__(self, n_components=2):
        '''
        Input: number of principle-components
        
        Constructor method for the Principle Component Analysis class
        '''
        self.n_components = n_components
    
    
    def fit(self, df):
        '''
        Input: data/dataframe
        
        Stores the input data that is needed to be transformed
        '''
        self.df = df
        self.num_samples, self.num_features = df.shape
        
        eig_vals, eig_vecs = self.eigen_values_and_vectors(self.df)
        self.eigmat = eig_vecs[:,:self.n_components]
        
    
    def mean_of_features(self, df):
        '''
        Input: data/dataframe
        
        Returns numpy array of the means of the features of the input
        '''
        _df = np.array(df)
        return np.mean(_df, axis=0)
    
    
    def std_of_features(self, df):
        '''
        Input: data/dataframe
        
        Returns the numpy array of the standard deviations of the features of the input
        '''
        _df = np.array(df)
        return np.std(_df, axis=0)

    
    def covariance_matrix(self, df):
        '''
        Input: data/dataframe
        
        Returns the covariance matrix of the input
        '''
        num_samples, num_features = df.shape
        
        _df = np.array(df)
        feature_means = self.mean_of_features(df)
        _df_transpose = _df.T
        
        for feature in range(num_features):
            _df_transpose[feature] = _df_transpose[feature] - feature_means[feature]
            
        cov_mat = np.zeros((num_features, num_features))
        
        for feature1 in range(num_features):
            for feature2 in range(num_features):
                cov_mat[feature1][feature2] = np.sum(_df_transpose[feature1]*_df_transpose[feature2])/(num_samples-1)
        
        return cov_mat
    
    
    def eigen_values_and_vectors(self, df):
        '''
        Input: data/dataframe
        
        Returns the (eigen values, eigen vectors) for the covariance matrix of the input
        '''
        eig_val, eig_vec = np.linalg.eig(self.covariance_matrix(df))
        
        # sorting the eigen values in descending order and corresponding order in eigen vectors
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        
        return eig_val, eig_vec
    
    
    def transform(self, df):
        '''
        Input: dataframe / dataset
        
        Returns transformed input as a pandas dataframe
        '''
        if len(df.shape) == 1:
            return np.dot(df, self.eigmat)
        
        transformed_df = pd.DataFrame(np.dot(df, self.eigmat))
        transformed_df.columns = [f'component-{num+1}' for num in range(self.n_components)]
       
        return transformed_df

    
    def plot_change_in_variance(self):
        '''
        Plots a bar graph to show the change in variance as you increase the number of components
        '''
        eig_vals, eig_vecs = self.eigen_values_and_vectors(self.df)
        
        explained_var_array = list()
        components_array = list()
        
        # num_components = len(eig_vals)
        num_components = self.n_components
        
        cummulative_sum = 0
        for each in range(len(eig_vals)):
            cummulative_sum += eig_vals[each]
            explained_var_array.append(cummulative_sum)
            
        explained_var_array = np.array(explained_var_array)/np.sum(eig_vals)
        
        fig = plt.figure(figsize=(10, 5))
        components_array = [each+1 for each in range(len(eig_vals))]
        
        plt.subplot(1, 2, 1)
        plt.bar(components_array, explained_var_array, alpha=0.5, label='cumulative explained variance')
        plt.xlabel('number of principle-components')
        plt.ylabel('cumulative explained variance')
        
        plt.subplot(1, 2, 2)
        plt.bar(components_array[:num_components], (eig_vals/np.sum(eig_vals))[:num_components], alpha=0.5, label='individual explained variance')
        plt.xlabel('number of principle-components')
        plt.ylabel('individual explained variance')
        
        plt.show()
        
        
    def scatter_plot(self):
        '''
        Plots a scatter plot to show the direction of the eigenvectors along with the data points
        '''
        eig_vals, eig_vecs = self.eigen_values_and_vectors(self.df)
        
        fig = plt.figure(figsize=(10,10))
        
        colors = ['y', 'm', 'r', 'g', 'b']
        
        # colors = []
        # for each in range(self.n_components):
            # r = random.randint(30,225)
            # g = random.randint(30,225)
            # b = random.randint(30,225)
            # colors.append((r,g,b))
            # colors.append("#"+''.join([random.choice('ABCDEF0123456789') for num in range(6)]))
        
        df = self.transform(self.df)
        comp1 = df['component-1']
        comp2 = df['component-2']
        plt.scatter(comp1, comp2, label='data points', alpha=0.5)
        
        h = 10
        x1 = abs(comp1.min())
        x2 = abs(comp1.max())
        x_range = max(x1, x2)*1.1
        
        y1 = abs(comp2.min())
        y2 = abs(comp2.max())
        y_range = max(y1, y2)*1.1
        
        for each in range(self.n_components):
            plt.quiver(0, 0, eig_vecs[each][0]*1e3, eig_vecs[each][1]*1e3, scale=5, label=f'eigen-vector-{each+1}', color=[colors[each]])
        
        plt.xlim(-x_range,x_range)
        plt.ylim(-y_range,y_range)
        plt.xlabel('component-1')
        plt.ylabel('component-2')
        plt.legend()
        plt.show()
        


# ### For standardized data

# In[43]:


std_pca = PCA(n_components=2)
std_pca.fit(X_std)
pca_std_data = std_pca.transform(X_std)
pca_std_data


# In[44]:


std_pca.plot_change_in_variance()


# In[45]:


std_pca.scatter_plot()


# ### For non-standardized data

# In[46]:


non_std_pca = PCA(n_components=2)
non_std_pca.fit(X_non_std)
pca_non_std_data = non_std_pca.transform(X_non_std)
pca_non_std_data


# In[47]:


non_std_pca.plot_change_in_variance()


# In[48]:


non_std_pca.scatter_plot()


# ## Training classification models and 5-fold cross-validation score

# In[49]:


model_names = ['RandomForestClassifier', 'BaggingClassifier', 'SVMClassifier']
models = [RandomForestClassifier(), BaggingClassifier(), SVC()]


# ### For standardized data

# In[50]:


cross_validation_plots(models, pca_std_data, y_std, model_names, title='for standardized data')


# ### For non-standardized data

# In[51]:


cross_validation_plots(models, pca_non_std_data, y_non_std, model_names, title='for non-standardized data')


# In[52]:


def evaluation_metrics_before_PCA(models, model_names, train_X, train_y, test_X, test_y):
    print('Before PCA:-')
    for each in range(len(models)):
        model = models[each]
        model.fit(train_X, train_y)
        print(f'for {model_names[each]}:-')
        pred_y = model.predict(test_X)
        target_names = [f'class-{Class}' for Class in np.unique(train_y)]
        score = model.score(test_X, test_y)
        print(f'Accuracy score: {score}')
        print(f'Classification-report:-')
        print(classification_report(test_y, pred_y, target_names=target_names))
        print('-'*100)


# In[53]:


def evaluation_metrics_after_PCA(models, model_names, train_X, train_y, test_X, test_y, n_components=3):
    print('After PCA:-')
    for each in range(len(models)):
        pca = PCA(n_components=n_components)
        pca.fit(train_X)
        pca_train_X = pca.transform(train_X)
        pca_test_X = pca.transform(test_X)

        model = models[each]
        model.fit(pca_train_X, train_y)
        print(f'for {model_names[each]}:-')
        pred_y = model.predict(pca_test_X)
        target_names = [f'class-{Class}' for Class in np.unique(train_y)]
        score = model.score(pca_test_X, test_y)
        print(f'Accuracy score: {score}')
        print(f'Classification-report:-')
        print(classification_report(test_y, pred_y, target_names=target_names))
        print('-'*100)


# ## for standardized data

# ### before PCA

# In[54]:


evaluation_metrics_before_PCA(models, model_names, train_X_std, train_y_std, test_X_std, test_y_std)


# ### after PCA

# In[55]:


evaluation_metrics_after_PCA(models, model_names, train_X_std, train_y_std, test_X_std, test_y_std)


# ## for non-standardized data

# ### before PCA

# In[56]:


evaluation_metrics_before_PCA(models, model_names, train_X_non_std, train_y_non_std, test_X_non_std, test_y_non_std)


# ### after PCA

# In[57]:


evaluation_metrics_after_PCA(models, model_names, train_X_non_std, train_y_non_std, test_X_non_std, test_y_non_std)


# ## Finding the optimal number of principal components

# ### (Completed previously)

# ---
# # Question-2
# ---

# In[58]:


wines_df = pd.read_csv('wine_data.csv')
wines_df


# In[59]:


wines_df.isnull().sum()


# In[60]:


wines_df.dtypes


# In[61]:


print_num_unique_entries(wines_df)


# In[62]:


fig = plt.figure(figsize=(15,10))
sns.heatmap(wines_df.corr(), annot=True)
plt.show()


# In[63]:


wines_df.hist(figsize=(20,20))
plt.show()


# In[64]:


wines_X = wines_df[wines_df.columns.tolist()[1:]].copy()
wines_y = wines_df['class'].copy()


# In[65]:


wines_X


# In[66]:


wines_y


# In[67]:


wines_X = scaler.fit_transform(wines_X)
wines_X = pd.DataFrame(wines_X)
wines_X.columns = wines_df.columns.tolist()[1:]


# In[68]:


wines_X


# In[69]:


wines_train_X, wines_test_X, wines_train_y, wines_test_y = train_test_split(wines_X, wines_y, random_state=0)


# In[70]:


class GaussianBayesClassifier:
    def __init__(self):
        pass
    
    def train(self, train_X, train_y):
        '''
        Input: x,y (training data)
        
        Trains the model
        '''
        self.classes = list(np.unique(train_y))
        self.num_classes = len(self.classes)
        self.num_datapoints, self.num_features = train_X.shape
        
        self.classwise_variances = {}
        for Class in self.classes:
            self.classwise_variances[Class] = np.array(train_X[train_y == Class].cov())

        self.classwise_means = {}
        for Class in self.classes:
            self.classwise_means[Class] = np.mean(train_X[train_y == Class])
        
        self.class_priors = dict()
        for k in self.classes:
            self.class_priors[k] = np.count_nonzero(train_y == k)/self.num_datapoints
        
        
    def test(self, test_X):
        '''
        Input: testing data
        Output: predictions for every instance in the testing data as a numpy array
        '''
        test_X = np.array(test_X)
        
        pred_y = []
        for x in test_X:
            pred_y.append(self.predict(x))
        
        pred_y = np.array(pred_y)
        return pred_y
        
        
    def predict(self, test_X):
        '''
        Input: a single data point
        Output: predicted class
        '''
        classwise_predictions = {}
        for Class in self.classes:
            classwise_predictions[Class] = self.gi(test_X, Class)
            
        argmax_class = max(zip(classwise_predictions.values(), classwise_predictions.keys()))[1]
        return argmax_class
          
    def predict_proba(self, test_X):
        '''
        Input: a single data point
        Output: prediction probability of each class as a dictionary for the input data point
        '''
        classwise_predictions = {}
        total_prob = 0
        for Class in self.classes:
            prob = math.exp(self.gi(test_X, Class))
            classwise_predictions[Class] = prob
            total_prob += prob
            
        for Class in self.classes:
            classwise_predictions[Class] /= total_prob
            
        return classwise_predictions

    
    def gi(self, x, Class):
        '''
        Input:-
            x : feature vector
            Class: class for which we need to find the value of the discriminant function
        Output:-
            Value of discriminant function for the given vector and class
            (ignoring unnecessary terms that are not used in comparisions)
        '''
        mu = self.classwise_means[Class].copy()
        sigma = self.classwise_variances[Class].copy()
        
        sigma_inv = np.linalg.inv(sigma)
        sigma_det = np.linalg.det(sigma)
            
        return -0.5*(x - mu).T@sigma_inv@(x - mu) - 0.5*np.log(sigma_det) + np.log(self.class_priors[Class])


# ## Linear Discriminant Analysis implementation from scratch

# In[71]:


class LDA:
    def __init__(self, n_components=None):
        '''
        Constructor for the Linear Discriminant Analysis Model
        
        n_components = number of components
        '''
        self.n_components = n_components
    
    
    def fit(self, train_X, train_y):
        '''
        Input: training dataset and labels
        
        Fits the dataset and labels into the model
        '''
        self.fit_X = np.array(train_X)
        self.fit_y = np.array(train_y)
        
        self.num_samples, self.num_features = train_X.shape
        self.feature_means = np.mean(train_X, axis=0)
                
        self.uniqueclasses = np.unique(train_y)
        self.classwise_features = dict()
        
        for Class in self.uniqueclasses:
            self.classwise_features[Class] = np.array(train_X[train_y == Class].copy())
            
        self.classwise_feature_means = dict()
        for Class in self.classwise_features.keys():
            self.classwise_feature_means[Class] = np.mean(self.classwise_features[Class], axis=0)
        
        self.classwise_cov_mats = dict()
        for Class in self.classwise_features.keys():
            feature_numpy = self.classwise_features[Class]
            feature_mean = self.classwise_feature_means[Class]
            
            cov_mat = np.zeros((self.num_features, self.num_features))
            
            for each in range(len(feature_numpy)):
                temp = feature_numpy[each] - feature_mean
                temp = temp.reshape((self.num_features, 1))
                cov_mat = cov_mat + np.dot(temp, temp.T)
                
            self.classwise_cov_mats[Class] = cov_mat
            
        self.create_within_class_scatter_matrix()
        self.create_between_class_scatter_matrix()
        
        eig_vals, eig_vecs = self.eigen_operations()
        if self.n_components is None:
            self.n_components = self.find_best_n_components()
        
        self.eigmat = eig_vecs[:,:self.n_components]
    
    
    def create_within_class_scatter_matrix(self):
        '''
        Creates the within-class scatter matrix
        '''
        self.SW = np.zeros(self.classwise_cov_mats[self.uniqueclasses[0]].shape)
        
        for Class in self.classwise_cov_mats.keys():
            self.SW = self.SW + self.classwise_cov_mats[Class]
    
    
    def create_between_class_scatter_matrix(self):
        '''
        Creates the between-class scatter matrix
        '''
        self.SB = np.zeros(self.SW.shape)
        
        self.all_features_mean = np.array(self.fit_X.mean())
        
        for Class in self.classwise_features.keys():
            temp = self.classwise_feature_means[Class] - self.all_features_mean
            temp = temp.reshape((self.num_features, 1))
            self.SB = self.SB + len(self.classwise_features[Class])*np.dot(temp, temp.T)
    
    
    def eigen_operations(self):
        '''
        Returns eigen values (in sorted order) and corresponding eigen vectors using the scatter matrices 
        '''
        sw_inv = np.linalg.inv(self.SW)
        
        eig_val, eig_vec = np.linalg.eig(np.dot(sw_inv, self.SB))
        
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        
        return eig_val, eig_vec
    
    
    def transform(self, df):
        '''
        Input: dataset / dataframe
        
        Returns the transformed dataframe after Linear Discriminant Analysis
        '''
        if len(df.shape) == 1:
            return np.real(np.dot(df, self.eigmat))
    
        transformed_df = pd.DataFrame(np.real(np.dot(df, self.eigmat)))
        transformed_df.columns = [f'component-{num+1}' for num in range(self.n_components)]

        return transformed_df

    
    def find_best_n_components(self, threshold=0.95):
        '''
        Input: threshold
        
        Automatically returns the best number of linear discriminants based upon
        the percentage of variance that needs to be conserved (threshold)
        '''
        eig_vals, eig_vecs = self.eigen_operations()
        
        current_threshold = 0
        best_n = 1
        
        eig_vals_sum = np.sum(eig_vals)
        eig_vals_cumsum = np.cumsum(eig_vals)
        
        for n in range(len(eig_vals)):
            if eig_vals_cumsum[n]/eig_vals_sum > threshold:
                break
            best_n += 1
        
        return best_n
    
    
    def euclidean_distance(self, point1, point2):
        '''
        Input: 2 data points
        
        Returns the euclidean distance between the inputs
        '''
        return np.linalg.norm(point1 - point2)
    
    
    def predict_single_datapoint(self, test_X):
        '''
        Input: a single test data point

        Predicts and returns the best matching class for the input
        '''
        transformed_datapoint = np.array(self.transform(test_X))
        
        best_class = self.uniqueclasses[0]
        shortest_distance =             self.euclidean_distance(transformed_datapoint, np.array(self.transform(self.classwise_feature_means[best_class])))
        
        for Class in self.classwise_feature_means.keys():
            if Class is not best_class:
                class_mean = np.array(self.transform(self.classwise_feature_means[Class]))
                dist = self.euclidean_distance(transformed_datapoint, class_mean)
                if dist < shortest_distance:
                    shortest_distance = dist
                    best_class = Class

        return best_class


    def predict(self, test_X):
        '''
        Input: test dataset / dataframe 
        
        Returns the model predictions for the input
        '''
        test_X = np.array(test_X)
        prediction_array = list()
        
        for each in test_X:
            prediction_array.append(self.predict_single_datapoint(each))
        
        return np.array(prediction_array)


    def predict_gnb(self, test_X):
        '''
        Input: test dataset / dataframe 
        
        Returns the model predictions for the input
        (based on Gaussian Naive Bayes)
        '''
        train_X = self.transform(self.fit_X)
        test_X = self.transform(test_X)
        train_y = self.fit_y
        
        model = GaussianBayesClassifier()
        model.train(train_X, train_y)

        return model.test(test_X)
       
        
    def predict_proba(self, test_X):
        '''
        Input: test dataset
        
        Returns an array of dictionaries with prediction probabilities of each class for each input data point
        '''
        train_X = self.transform(self.fit_X)
        test_X = np.array(self.transform(test_X))
        train_y = self.fit_y
        
        model = GaussianBayesClassifier()
        model.train(train_X, train_y)
        
        prob_arr = list()
        for each in range(len(test_X)):
            prob_arr.append(model.predict_proba(test_X[each]))

        return prob_arr
            
        
#     def scores_transform(self):
#         '''
#         Returns the score column for the training dataset
#         '''
#         eig_vals, eig_vecs = self.eigen_operations()
#         transformed_data = np.dot(self.fit_X, eig_vecs)
        
#         return np.real(np.sum(transformed_data, axis=1))
    
    
#     def gaussian(self, x, _mean, _std):
#         '''
#         Returns probability based on Gaussian Bayes Distribution
#         '''
#         return (1/math.sqrt(2*math.pi))*math.exp(-0.5*((x-_mean)/_std)**2)/_std
    
    
#     def proba(self):
#         '''
#         Returns array of probabilities corresponding to each row to belong to its corresponding class 
#         '''
#         scores_df = pd.DataFrame(self.scores_transform())
#         scores_np = np.array(scores_df)
#         scores_df.columns = ['scores']
#         scores_df['class'] = self.fit_y
        
#         scores_means = dict()
#         scores_stds = dict()
        
#         for Class in self.uniqueclasses:
#             class_df = scores_df['scores'][scores_df['class']==Class]
#             mean = float(np.mean(class_df))
#             std = float(np.std(class_df))
            
#             scores_means[Class] = mean
#             scores_stds[Class] = std
        
#         proba_list = list()
#         for each in range(len(scores_np)):
#             x = float(scores_np[each])
#             Class = self.fit_y[each]
#             proba_list.append(self.gaussian(x, scores_means[Class], scores_stds[Class]))
        
#         proba_list = np.array(proba_list)
#         return np.array(proba_list)
    
        
    def score(self, test_X, test_y):
        '''
        Input: test data and labels
        
        Returns the accuracy score of the model on the input
        '''
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        
        pred_y = self.predict_gnb(test_X)
        
        acc_score = 0
        
        for each in range(len(test_X)):
            if(pred_y[each] == test_y[each]):
                acc_score += 1
                
        acc_score /= len(test_X)
        return acc_score
    
    
    def scatter_plot_for_best_features(self):
        '''
        Scatter plot to visualize the feature space for the
        features that have a high impact on the classification tasks
        '''
        transformed_data = self.transform(self.fit_X)
        
        fig = plt.figure(figsize=(7,7))
        
        colors = ['c', 'm', 'y', 'b', 'g', 'r']
        
        color_counter = 0
        for each in self.uniqueclasses:
            x = transformed_data['component-1'][self.fit_y==each]
            y = transformed_data['component-2'][self.fit_y==each]
            plt.scatter(x, y, color=colors[color_counter], label=f'class-{each}')
            color_counter += 1
        plt.xlabel('component-1')
        plt.ylabel('component-2')
        plt.legend()
        plt.show()
        
        
    def decision_boundary_plot(self, model, X, y):
        '''
        Plots the scatter plot of any two features among the features 
        which contribute to the maximum variance the decision boundary
        '''
        train_X = self.transform(X)
        train_y = y
        
        model.fit(train_X, train_y)
        
        h = 0.02
        
        xf1 = train_X['component-1'].to_numpy()
        xf2 = train_X['component-2'].to_numpy()
        train_y = train_y.to_numpy()

        x_min, x_max = xf1.min() - 10*h, xf1.max() + 10*h
        y_min, y_max = xf2.min() - 10*h, xf2.max() + 10*h
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = plt.figure(figsize=(5,5))
        plt.contourf(xx, yy, Z, alpha=0.25)
        plt.contour(xx, yy, Z, colors='k', linewidths=0.7)

        plt.scatter(xf1, xf2, c=train_y, edgecolors='k')
        plt.xlabel('component-1')
        plt.ylabel('component-2')
        plt.show()
        


# In[72]:


lda_scratch = LDA()
lda_scratch.fit(wines_train_X, wines_train_y)
lda_scratch.transform(wines_train_X)


# In[73]:


lda_scratch.predict_gnb(wines_test_X)


# In[74]:


lda_scratch.score(wines_test_X, wines_test_y)


# In[75]:


lda_scratch.predict_proba(wines_test_X)


# In[76]:


lda_scratch.scatter_plot_for_best_features()


# In[77]:


lda_scratch.decision_boundary_plot(RandomForestClassifier(), wines_train_X, wines_train_y)


# In[78]:


lda_scratch.decision_boundary_plot(BaggingClassifier(), wines_train_X, wines_train_y)


# In[79]:


lda_scratch.decision_boundary_plot(SVC(), wines_train_X, wines_train_y)


# In[80]:


comp_models = [RandomForestClassifier(), BaggingClassifier(), SVC()]
comp_model_names = ['RandomForestClassifier', 'BaggingClassifier', 'SVMClassifier']


# In[81]:


def pca_vs_lda(n_components, models, model_names, train_X, train_y, test_X, test_y):
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(train_X)
    pca_train_X = pca.transform(train_X)
    pca_test_X = pca.transform(test_X)

    # LDA
    lda = LDA(n_components=None)
    lda.fit(train_X, train_y)
    lda_train_X = lda.transform(train_X)
    lda_test_X = lda.transform(test_X)
    
    # training the classifiers
    for each in range(len(models)):
        print(f'Classifier: {model_names[each]}')
        modelpca = models[each].fit(pca_train_X, train_y)
        modellda = models[each].fit(lda_train_X, train_y)
        print(f'Accuracy score for PCA: {modelpca.score(pca_test_X, test_y)}')
        print(f'Accuracy score for LDA: {modellda.score(lda_test_X, test_y)}')
        print()


# In[82]:


pca_vs_lda(2, comp_models, comp_model_names, wines_train_X, wines_train_y, wines_test_X, wines_test_y)


# ## 5-fold cross-validation, ROC plotting, and AUC calculation from scratch 

# In[83]:


def prob_vector(model, test_X, Class):
    probas = model.predict_proba(test_X)
    return np.array([probas[each][Class] for each in range(len(probas))])


# In[84]:


def thresholds(num):
    return np.array([each/num for each in range(num+1)])


# In[85]:


def plot_ROC(y_test, y_pred_proba, num_thresholds=100):
    roc_x, roc_y = [1], [1]
    
    idx = y_pred_proba.argsort()[::-1]
    y_pred_proba = y_pred_proba[idx]
    y_test = y_test[idx]
    
    thresh = thresholds(num_thresholds)
    for th in range(num_thresholds):
        tp, fp, tn, fn = 0, 0, 0, 0
        for each in range(len(y_test)):
            actual_class = y_test[each]
            
            predicted_class = 0 
            if y_pred_proba[each] >= thresh[th]:
                predicted_class = 1
                
            if actual_class==1 and predicted_class==1:
                tp += 1
            elif actual_class==1 and predicted_class==0:
                fn += 1
            elif actual_class== 0 and predicted_class==1:
                fp += 1
            else:
                tn += 1
            
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        
        roc_x.append(fpr)
        roc_y.append(tpr)
            
    roc_x.append(0)
    roc_y.append(0)
            
    roc_x = roc_x[::-1]
    roc_y = roc_y[::-1]
        
    auc = abs(np.trapz(roc_y, roc_x))
    
    plt.plot(roc_x, roc_y, label=f'AUC = {auc}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()


# In[86]:


def Kfold_scratch(X, y, n_splits=5):
    model = LDA()
    
    df = pd.DataFrame(X)
    df['y'] = y
    df = np.array(df)
    np.random.shuffle(df)
    
    unique_classes = np.unique(y)
    
    num_samples = df.shape[0]
    indices = list()
    len_split = num_samples//n_splits
    for each in range(0, num_samples+1, len_split):
        indices.append(each)
    
    indices[-1] = num_samples
        
    folds_X = list()
    folds_y = list()
    
    for ind in range(n_splits):
        curr_ind = indices[ind]
        next_ind = indices[ind+1]
        folds_X.append(df[curr_ind:next_ind, :-1])
        folds_y.append(df[curr_ind:next_ind, -1])
    
    cross_validation_scores = list()
    
    for ind in range(len(folds_X)):
        temp = list()
        test_X, test_y = folds_X[ind], folds_y[ind]
        train_X, train_y = 0, 0
        check = 0
        
        for each in range(len(folds_X)):
            if each != ind:
                if check == 0:
                    train_X = folds_X[each]
                    train_y = folds_y[each]
                    check = 1
                else:
                    train_X = np.vstack(folds_X[each])
                    train_y = np.vstack(folds_y[each])[:, 0]
        
        model.fit(train_X, train_y)
        cross_validation_scores.append(model.score(test_X, test_y))
        
        for pos_class in unique_classes:
            print(f'ROC curve for class-{pos_class}')
            copy_test_y = np.array(test_y)
            copy_pred_y = prob_vector(model, test_X, pos_class)
            for each in range(len(test_y)):
                if copy_test_y[each] == pos_class:
                    copy_test_y[each] = 1
                else:
                    copy_test_y[each] = 0

            plot_ROC(copy_test_y, copy_pred_y)
        print('-'*100)
        print()
    
    cross_validation_scores = np.array(cross_validation_scores)
    
    print(f'individual cross-validation_scores: {cross_validation_scores}')
    print(f'average cross-validation score: {np.mean(cross_validation_scores)}')
    print()


# In[87]:


Kfold_scratch(wines_X, wines_y)

