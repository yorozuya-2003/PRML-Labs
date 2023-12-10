#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import fetch_olivetti_faces, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier


# ---
# # Q1
# ---

# ## Preprocessing and Analysis

# In[3]:


df = pd.read_csv("glass.csv")
df = df[df.columns[1:]]
df


# In[4]:


df.isnull().sum()


# In[5]:


for each in df.columns:
    print('Number of unique entries in', each, ':', len(df[each].unique()))


# In[6]:


df.describe()


# In[7]:


features = df.columns.tolist()[:-1]


# In[8]:


X = df[features]
y = df['Type']


# In[9]:


X.hist(figsize=(15,15))
plt.show()


# In[10]:


y.hist(figsize=(5,5))
plt.show()


# In[11]:


sns.pairplot(df, hue='Type')
plt.show()


# In[12]:


figure = plt.figure(figsize=(10,8))
heatmap = sns.heatmap(df.corr(), annot=True)
plt.show()


# In[13]:


def normalize(df, feature):
    f_mean = df[feature].mean()
    f_std = df[feature].std()
    df[feature] = (df[feature]-f_mean)/f_std
    return df


# In[14]:


for each in features:
    X = normalize(X, each)


# In[15]:


X = pd.DataFrame(X)
X


# ## Implementing K-Means clustering algorithm on the dataset & Visualization

# In[16]:


def kmeans_cluster_plot(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    label = kmeans.fit_predict(X)
    uniquelabels = np.unique(label)

    fig = plt.figure(figsize=(8, 5))
    
    Xnp = X.to_numpy()
    
    for each in uniquelabels:
        plt.scatter(Xnp[label == each, 0] , Xnp[label == each , 1] , label = each, alpha=0.7)
    plt.legend()

    centers = np.array(kmeans.cluster_centers_)
    plt.scatter(centers[:,0], centers[:,1], marker="x", color='black')
    plt.title(f'Number of clusters: {n_clusters}')

    plt.show()


# In[17]:


for each in range(2, 8):
    kmeans_cluster_plot(X, n_clusters=each)


# ## Finding optimal value of k using the Silhouette score

# In[18]:


def kmeans_silhouette_score(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    label = kmeans.fit_predict(X)
    
    sscore = silhouette_score(X, label)
    
    return sscore


# In[19]:


n_clusters_range = 15
n_clusters_array = []
sscore_array = []

best_sscore = 0
best_n_clusters = 2

for each in range(2, n_clusters_range+1):
    sscore = kmeans_silhouette_score(X, n_clusters=each)
    
    n_clusters_array.append(each)
    sscore_array.append(sscore)
    
    if sscore > best_sscore:
        best_sscore = sscore
        best_n_clusters = each
        
    print(f'silhouette score for n_clusters={each} is {sscore}')


# In[20]:


fig = plt.figure(figsize=(10,6))
plt.plot(n_clusters_array, sscore_array, 'o-')
plt.show()


# In[21]:


best_n_clusters, best_sscore


# ## Elbow Method to find the optimal k value for k-means algorithm

# In[22]:


fig = plt.figure(figsize=(10,6))
# within clusters sum of squares
wcss = []
x_elbow = []

for n_clusters in range(2, 11):
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    wcss.append(clustering.inertia_)
    x_elbow.append(n_clusters)
    
plt.plot(x_elbow, wcss, 'o-')
plt.show()


# ### Bagging with the KNN classifier as the base model

# In[23]:


train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)


# In[24]:


def AccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(train_X, train_y)
    print('training accuracy:', knn_clf.score(train_X, train_y))
    print('testing accuracy:', knn_clf.score(test_X, test_y))

def BaggingAccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors):
    bagging_clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=n_neighbors), random_state=0)
    bagging_clf.fit(train_X, train_y)
    print('training accuracy:', bagging_clf.score(train_X, train_y))
    print('testing accuracy:', bagging_clf.score(test_X, test_y))


# ### KNN Accuracies (without Bagging)

# In[25]:


AccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors=1)


# In[26]:


AccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors=2)


# In[27]:


AccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors=3)


# ### Bagging Accuracies

# In[28]:


BaggingAccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors=1)


# In[29]:


BaggingAccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors=2)


# In[30]:


BaggingAccuracyKNN(train_X, train_y, test_X, test_y, n_neighbors=3)


# ---
# # Q2
# ---

# In[31]:


from collections import Counter


# ## Implementation of K-Means Algorithm from scratch

# In[32]:


class KMeansAlgo:
    def __init__(self, k=8, max_iter=300, user_centroids=None, random_state=None):
        '''
        Inputs:-
        k = number of clusters
        max_iter = maximum number of iterations
        user_centroids = custom cluster centroids initialised by the user
        
        Constructor method for the K-Means Algorithm
        Also initialises the cluster centroids, current iteration number
        '''
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        
        if user_centroids is None:
            self.cluster_centroids = None
        else:
            self.cluster_centroids = dict()
            for z in range(k):
                self.cluster_centroids[z] = user_centroids[z]
            
        self.current_iter = 0
        self.sse_error = 0
    
    
    def fit(self, train_X):
        '''
        Input: training dataset
        
        Trains the clustering model by manipulating and storing the cluster centers
        based on the number of iterations and comparision of euclidean distances of the datapoints
        '''
        train_X = train_X.to_numpy()
        train_X_df = pd.DataFrame(train_X)
        
        prev_clusters = dict()
        prev_centroids = dict()
        prev_sse_error = None
        
        if self.cluster_centroids is None:
            centroids = train_X_df.sample(n=self.k, random_state=self.random_state).to_numpy()
            self.cluster_centroids = dict()
            for z in range(self.k):
                self.cluster_centroids[z] = centroids[z]
            
        while not self.convergence():
            self.clusters = dict()
            for z in range(self.k):
                self.clusters[z] = list()

            for datapoint in train_X:
                cluster = self.compare_distances(datapoint)
                self.clusters[cluster].append(datapoint)
                
            for each in range(self.k):
                self.cluster_centroids[each] = np.mean(self.clusters[each], axis=0)
            
            self.sse_error = self.sse()
            
            if self.current_iter != 0:
                '''
                if self.sse_error == prev_sse_error:
                    print(f'sse not changing, converged after {self.current_iter+1} iterations')
                    break
                '''
                #'''         
                check = self.k
                for each in range(self.k):
                    if Counter(self.cluster_centroids[each]) == Counter(prev_centroids[each]):
                        check -= 1
                if check == 0:
                    print(f'centroids not changing, converged after {self.current_iter+1} iterations')
                    break
                #'''
                    
            prev_clusters = self.clusters
            prev_centroids = self.cluster_centroids.copy()
            prev_sse_error = self.sse_error.copy()
            
            self.current_iter += 1
        
        self.cluster_centroids = prev_centroids
        self.sse_error = prev_sse_error
    
    
    def compare_distances(self, datapoint):
        '''
        Input: datapoint
        
        Compares the euclidean distances of the datapoint from all the cluster centroids
        Returns the class of the cluster from the centroid of which the euclidean distance is the minimum
        '''
        distances = dict()
        
        for each in self.cluster_centroids.keys():
            distances[each] = self.euclidean_distance(datapoint, self.cluster_centroids[each])
        
        keymin = min(zip(distances.values(), distances.keys()))[1]
        return keymin
    
    
    def euclidean_distance(self, datapoint, centroid):
        '''
        Input: datapoint, centroid datapoint
        
        Returns the euclidean distance of the datapoint from the centroid
        '''
        return np.linalg.norm(np.array(datapoint) - np.array(centroid))


    def convergence(self):
        '''
        Checks the condition of convergence
        (maximum number of iterations is reached)
        (implemented in the fit method: cluster centers are not changing anymore)
        '''
        if self.current_iter >= self.max_iter - 1:
            return True
        return False
    
    
    def sse(self):
        '''
        Returns Sum of Sqaured Error
        '''
        error = 0
        for each in range(self.k):
            for datapoint in self.clusters[each]:
                error += self.euclidean_distance(datapoint, self.cluster_centroids[each])**2
        
        return error
    
    
    def predict(self, test_X):
        '''
        Input: single test datapoint
        
        Returns the class of the cluster to which the datapoint belongs
        (based on the distances from the cluster centroids)
        '''
        return self.compare_distances(test_X)


# In[33]:


olivetti_dataset = fetch_olivetti_faces()


# In[34]:


olivetti_X = olivetti_dataset.data


# In[35]:


olivetti_y = olivetti_dataset.target


# In[36]:


olivetti_X.shape


# In[37]:


olivetti_y.shape


# In[38]:


olivetti_X = pd.DataFrame(olivetti_X)


# In[39]:


olivetti_df = olivetti_X.copy()
olivetti_df['label'] = olivetti_y


# In[40]:


olivetti_df


# ## Training the implemened model by selecting 40 random 4096 dimensional points as initializations

# In[41]:


random_points = olivetti_X.sample(n=40, random_state=0).to_numpy()


# In[42]:


scratch1 = KMeansAlgo(k=40, max_iter=300, user_centroids=random_points, random_state=0)
scratch1.fit(olivetti_X)


# In[43]:


scratch1.sse_error


# In[44]:


scratch1_clusters_dict = {}
for num in range(scratch1.k):
    scratch1_clusters_dict[f'Cluster-{num+1}'] = len(scratch1.clusters[num])
    print(f'Number of points in cluster-{num+1}:', len(scratch1.clusters[num]))
print()
print(scratch1_clusters_dict)


# In[45]:


fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
axarr = axarr.flatten()
counter = 0
for each in range(scratch1.k):
    axarr[counter].imshow((scratch1.cluster_centroids[each].reshape(64,64)))
    axarr[counter].set_title(f'cluster: {each+1}')
    counter += 1                 
plt.show()


# In[46]:


for num in range(scratch1.k):
    fig, axarr=plt.subplots(nrows=1, ncols=10, figsize=(18, 2))
    axarr=axarr.flatten()
    counter = 0
    for each in range(len(scratch1.clusters[num])):
        axarr[counter].imshow((scratch1.clusters[num][each].reshape(64,64)))
        counter += 1
        if(counter > 9):
            break
    plt.suptitle(f'Cluster: {num+1}')
    plt.show()


# ## Training the implemented model by selecting 40 images from each class as initializations

# In[47]:


random_points = []
for num in range(40):
    random_points.append(olivetti_X[olivetti_df['label']==num].sample(n=1, random_state=0).to_numpy())


# In[48]:


scratch2 = KMeansAlgo(k=40, max_iter=300, user_centroids=random_points, random_state=0)
scratch2.fit(olivetti_X)


# In[49]:


scratch2.sse_error


# In[50]:


scratch2_clusters_dict = {}
for num in range(scratch2.k):
    scratch2_clusters_dict[f'Cluster-{num+1}'] = len(scratch2.clusters[num])
    print(f'Number of points in cluster-{num+1}:', len(scratch2.clusters[num]))
print()
print(scratch2_clusters_dict)


# In[51]:


fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
axarr = axarr.flatten()
counter = 0
for each in range(scratch2.k):
    axarr[counter].imshow((scratch2.cluster_centroids[each].reshape(64,64)))
    axarr[counter].set_title(f'cluster: {each+1}')
    counter += 1                 
plt.show()


# In[52]:


for num in range(scratch2.k):
    fig, axarr=plt.subplots(nrows=1, ncols=10, figsize=(18, 2))
    axarr=axarr.flatten()
    counter = 0
    for each in range(len(scratch2.clusters[num])):
        axarr[counter].imshow((scratch2.clusters[num][each].reshape(64,64)))
        counter += 1
        if(counter > 9):
            break
    plt.suptitle(f'Cluster: {num+1}')
    plt.show()


# ---
# # Q3
# ---

# ## Preprocessing and Data Analysis

# In[53]:


customers_df = pd.read_csv('Wholesale customers data.csv')


# In[54]:


customers_df


# In[55]:


customers_df.columns.tolist()


# In[56]:


customers_df.isnull().sum()


# In[57]:


for each in customers_df.columns:
    print('Number of unique entries in', each, ':', len(customers_df[each].unique()))


# In[58]:


customers_df.hist(figsize=(15,15))
plt.show()


# In[59]:


sns.pairplot(customers_df)
plt.show()


# In[60]:


customers_noncat_features = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
customers_cat_features = ['Channel', 'Region']


# In[61]:


for each in customers_noncat_features:
    customers_df = normalize(customers_df, each)


# In[62]:


for each in customers_cat_features:
    customers_df = normalize(customers_df, each)


# ## Covariance Heatmap

# In[63]:


figure = plt.figure(figsize=(10,8))
heatmap = sns.heatmap(customers_df.cov(), annot=True)
plt.show()


# ## Visualization for the outliers

# In[64]:


def mahalanobis_distance(df):
    mu = df - np.mean(df)
    inv_covmat = np.linalg.inv(np.cov(df.values.T))
    md = np.dot(mu, inv_covmat)
    md = np.dot(md, mu.T)
    return md.diagonal()


# In[65]:


customers_df['mahalanobis distance'] = mahalanobis_distance(customers_df)


# In[66]:


customers_df


# In[67]:


threshold = np.mean(customers_df['mahalanobis distance']) * 0.5


# In[68]:


features = customers_df.columns.tolist()[:-1]


# In[69]:


for f in features:
    x = [x for x in range(len(customers_df[f]))]
    y = customers_df[f].to_numpy()
    t = threshold*np.ones(len(customers_df[f]))
    plt.scatter(x, y)
    plt.plot(x, t, 'r')
    plt.title(f'for feature: {f}')
    plt.show()


# ## Outliers Visualization

# In[70]:


customers_df['outliers'] = customers_df['mahalanobis distance'] > threshold


# In[71]:


outliers = customers_df[customers_df['outliers']==True].copy()
non_outliers = customers_df[customers_df['outliers']==False].copy()

plt.scatter(outliers['Grocery'], outliers['Detergents_Paper'], label='outliers')
plt.scatter(non_outliers['Grocery'], non_outliers['Detergents_Paper'], label='non-outliers')
plt.legend()
plt.xlabel('Grocery')
plt.ylabel('Detergents_Paper')

plt.show()


# ## DBSCAN clustering on the dataset

# In[72]:


clustering = DBSCAN().fit(customers_df[features])
labels = clustering.labels_
uniquelabels = np.unique(labels)
for each in uniquelabels:
    plt.scatter(x=customers_df['Grocery'].to_numpy()[labels==each], y=customers_df['Detergents_Paper'].to_numpy()[labels==each], label=each, alpha=0.5)
    plt.xlabel('Grocery')
    plt.ylabel('Detergents_Paper')
plt.legend()
plt.show()


# ## KMeans on the dataset

# In[73]:


customers_df = pd.read_csv('Wholesale customers data.csv')
for each in customers_noncat_features:
    customers_df = normalize(customers_df, each)


# In[74]:


customers_X = customers_df[['Channel','Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']].copy()
customers_y = customers_df['Region'].copy()


# In[75]:


kmeans = KMeans(n_clusters=2).fit(customers_X)
uniquelabels = np.unique(kmeans.predict(customers_X))
for each in uniquelabels:
    plt.scatter(x=customers_df['Grocery'].to_numpy()[labels==each], y=customers_df['Detergents_Paper'].to_numpy()[labels==each], label=each, alpha=0.5)
    plt.xlabel('Grocery')
    plt.ylabel('Detergents_Paper')
plt.legend()
plt.show()


# ## DBSCAN & KMeans clustering on the make_moons dataset

# In[76]:


moons_X, moons_y = make_moons(n_samples=2000, noise=0.2)


# In[77]:


moons_X = pd.DataFrame(moons_X)
moons_X.columns = ['feature-1', 'feature-2']


# In[78]:


def dbscan_moons_plot(moons_X=moons_X, eps=0.5, min_samples=5):
    moons_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(moons_X)
    labels = moons_dbscan.labels_
    uniquelabels = np.unique(labels)
    for each in uniquelabels:
        plt.scatter(x=moons_X['feature-1'].to_numpy()[labels==each], y=moons_X['feature-2'].to_numpy()[labels==each], label=each, alpha=0.5)
        plt.xlabel('feature-1')
        plt.ylabel('feature-2')
    plt.title(f'for eps={round(eps, 2)}, min_samples={min_samples}')
    plt.legend()
    plt.show()


# ### Hit-&-Trial Loop

# In[79]:


for e in range(2,6):
    for ms in range(5, 51, 5):
        dbscan_moons_plot(eps=0.05*e, min_samples=ms)


# ---

# ## DBSCAN Visualization

# In[80]:


dbscan_moons_plot(eps=0.2, min_samples=50)


# ---

# In[81]:


def kmeans_moons_plot(moons_X=moons_X, n_clusters=8):
    moons_kmeans = KMeans(n_clusters=n_clusters).fit(moons_X)
    label = moons_kmeans.predict(moons_X)
    uniquelabels = np.unique(label)
    for each in uniquelabels:
        plt.scatter(moons_X['feature-1'].to_numpy()[label==each], y=moons_X['feature-2'].to_numpy()[label==each], label=each, alpha=0.5)
        plt.xlabel('feature-1')
        plt.ylabel('feature-2')
    plt.legend()
    plt.title(f'for n_clusters={n_clusters}')
    plt.show()


# ### Hit-&-Trial Loop

# In[82]:


for n in range(2, 10):
    kmeans_moons_plot(n_clusters=n)


# ---

# ## KMeans Visualization

# In[83]:


kmeans_moons_plot(n_clusters=2)


# ---
