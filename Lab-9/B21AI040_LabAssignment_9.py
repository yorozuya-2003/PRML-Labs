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
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[3]:


import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms


# In[4]:


import pickle


# ---
# # Question-1
# ---

# ## Loading the MNIST dataset and performing train-validation-test split

# In[5]:


train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomCrop(size=28, padding=2),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])


# The values 0.1307 and 0.3081 used for the Normalize() transformation are the global mean and standard deviation of the MNIST dataset.

# In[6]:


mnist_dataset = datasets.MNIST('', train=True, download=True, transform=None)
mnist_dataset


# In[7]:


mnist_testset = datasets.MNIST('', train=False, download=True, transform=test_transform)
mnist_testset


# ### Train-validation-test split

# In[8]:


mnist_trainset, mnist_valset = random_split(mnist_dataset, [50000, 10000])


# In[9]:


len(mnist_trainset), len(mnist_valset)


# In[10]:


mnist_trainset[0][0]


# In[11]:


mnist_trainset[0][1]


# In[12]:


mnist_testset[0][0]


# In[13]:


mnist_testset[0][1]


# ### Applying augmentations to images

# In[14]:


mnist_trainset = list(mnist_trainset)
mnist_valset = list(mnist_valset)


# In[15]:


for each in range(len(mnist_trainset)):
    mnist_trainset[each] = (train_transform(mnist_trainset[each][0]), mnist_trainset[each][1])


# In[16]:


for each in range(len(mnist_valset)):
    mnist_valset[each] = (test_transform(mnist_valset[each][0]), mnist_valset[each][1])


# In[17]:


mnist_trainset[0][0].shape, mnist_trainset[0][0].dtype


# In[18]:


mnist_valset[0][0].shape, mnist_valset[0][0].dtype


# In[19]:


mnist_testset[0][0].shape, mnist_testset[0][0].dtype


# ## Plotting few images from each class

# In[20]:


images_dict = {}
for each in range(0, 10):
    images_dict[each] = []

img_counter = 0

for each in range(len(mnist_trainset)):
    if img_counter > 50:
        break
    img = mnist_trainset[each][0]
    label = mnist_trainset[each][1]
    if len(images_dict[label]) < 5:
        images_dict[label].append(img)
        img_counter += 1


# In[21]:


counter=0
for num in images_dict.keys():
    fig = plt.figure(figsize=(10,2))
    for each in range(1, 5+1):
        plt.subplot(1, 5, each)
        plt.imshow(images_dict[num][each-1].permute(1,2,0), cmap='bone')
        plt.title(f'label:{num}')
    counter+=1
    plt.show()


# ## Data loader for training set and testing set

# In[22]:


train_loader = DataLoader(dataset=mnist_trainset, batch_size=64, shuffle=False)
val_loader = DataLoader(dataset=mnist_valset, batch_size=1000, shuffle=False)
test_loader = DataLoader(dataset=mnist_testset, batch_size=1000, shuffle=False)


# ## 3-Layer MLP (all using linear layers)

# In[23]:


class MLP_PyTorch(nn.Module):
    def __init__(self, num_input_nodes, num_hidden_nodes_1, num_hidden_nodes_2, num_output_nodes):
        '''
        Input: numbers of nodes in the layers
        
        Constructor
        '''
        super().__init__()
        self.hidden_layer_1 = nn.Linear(num_input_nodes, num_hidden_nodes_1)
        self.hidden_layer_2 = nn.Linear(num_hidden_nodes_1, num_hidden_nodes_2)
        self.output_layer = nn.Linear(num_hidden_nodes_2, num_output_nodes)
        
        print('Number of trainable parameters:-')
        for param in self.parameters():
            print(f'{param.numel()}')

    
    def forward(self, x):
        '''
        Input: input layer dataset
        
        Returns the result of forward propagation by the model on the input
        '''
        x = x.reshape(x.shape[0], -1)
        x = relu(self.hidden_layer_1(x))
        x = relu(self.hidden_layer_2(x))
        x = self.output_layer(x)
        
        return x

    
    def save(self, path='best_model.pth'):
        '''
        Saves the model
        '''
        torch.save(self.state_dict(), path)

        
    def load(self, path='best_model.pth'):
        '''
        Loads the model
        '''
        self.load_state_dict(torch.load(path))
        self.eval()
        
        
    def fit(self, train_loader, val_loader,
              optimizer=Adam, loss_function=CrossEntropyLoss(), learning_rate=0.005, epochs=5):
        '''
        Input: Dataloader of the training dataset,
               Dataloader of the validation dataset
               optimizer (default = Adam),
               loss function (default = Cross Entropy loss),
               learning rate for the model (default = 0.005),
               number of epochs (default = 5)
        
        Trains the model, displays the training loss and
        the validation accuracy after each epoch, and
        also saves the best model at the end of each epoch
        '''
        best_accuracy = 0
        
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.loss_func = loss_function
        self.epochs = epochs
        
        self.loss_epochs = list()
        self.acc_epochs = list()
        self.epoch_arr = [each+1 for each in range(self.epochs)]
        
        for epoch in range(self.epochs):
            for batch in train_loader:
                train_X, train_y = batch
                pred_train_y = self.forward(train_X)

                loss = self.loss_func(pred_train_y, train_y)

                # applying backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # checking model performance on the validation set
            total_val_data = len(val_loader.dataset)
            correct_predictions = 0
            for batch in val_loader:
                val_X, val_y = batch
                val_pred_y = self.forward(val_X)
                for each in range(len(val_y)):
                    pred = int(torch.argmax(val_pred_y[each]))
                    true = int(val_y[each])
                    if pred == true:
                        correct_predictions += 1      

            val_acc = correct_predictions / total_val_data

            # saving the best model
            if val_acc > best_accuracy:
                self.save()

            # printing epoch-loss and validation-accuracy of the model
            epoch_loss = loss.item()
    
            self.loss_epochs.append(epoch_loss)
            self.acc_epochs.append(val_acc)
            
            print(f'epoch: {epoch+1}, loss: {epoch_loss}')
            print(f'validation accuracy: {val_acc}')
            print('-'*100)
            
        self.epochs_graphs()
            
            
    def test(self, test_loader):
        '''
        Input: Dataloader for the testing dataset
        
        Displays the model accuracy on the input
        '''
        # loading the best model
        self.load()
        
        total_test_data = len(test_loader.dataset)
        
        self.test_plot_correct = list()
        self.test_plot_incorrect = list()
        self.test_plot_arr_correct = list()
        self.test_plot_arr_incorrect = list()
        corr_plot_counter = 10
        incorr_plot_counter = 10 
        
        correct_predictions = 0
        for batch in test_loader:
            test_X, test_y = batch
            test_pred_y = self.forward(test_X)

            for each in range(len(test_y)):
                pred = int(torch.argmax(test_pred_y[each]))
                true = int(test_y[each])
                if pred == true:
                    correct_predictions += 1
                    if corr_plot_counter > 0:
                        self.test_plot_correct.append(test_X[each])
                        self.test_plot_arr_correct.append({'actual':true, 'predicted':pred})
                        corr_plot_counter -= 1
                else:
                    if incorr_plot_counter > 0:
                        self.test_plot_incorrect.append(test_X[each])
                        self.test_plot_arr_incorrect.append({'actual':true, 'predicted':pred})
                        incorr_plot_counter -= 1

        test_acc = correct_predictions / total_test_data
        print(f'Testing accuracy: {test_acc}')
        print('-'*100)
        
        self.test_graph()

        
    def epochs_graphs(self):
        '''
        Visualizes the loss-epoch and accuracy-epoch graphs for both training and validation
        '''
        figure = plt.figure(figsize=(12, 4))
        
        plt.subplot(1,2,1)
        plt.plot(self.epoch_arr, self.loss_epochs, 'o-', color='lightseagreen')
        plt.ylabel('epoch-training-loss')
        plt.xlabel('number of epochs')
        
        plt.subplot(1,2,2)
        plt.plot(self.epoch_arr, self.acc_epochs, 'o-', color='hotpink')
        plt.ylabel('epoch-validation-accuracy')
        plt.xlabel('number of epochs')
        plt.show()
        
        
    def test_graph(self):
        '''
        Visualizes the correct and incorrect predictions on the test data
        '''
        print('some of the correct predictions in the testing dataset:-')
        for num in range(2):
            fig = plt.figure(figsize=(10,2))
            for each in range(num*5, 5*(num+1)):
                plt.subplot(1, 5, each+1-(num*5))
                plt.imshow(self.test_plot_correct[each].permute(1,2,0), cmap='bone')
                Title = f"actual label:{self.test_plot_arr_correct[each]['actual']}\n"                     f"predicted label:{self.test_plot_arr_correct[each]['predicted']}"
                plt.title(Title, color='springgreen')
            plt.show()
            
        print('-'*100)
        print('some of the incorrect predictions in the testing dataset:-')
        
        for num in range(2):
            fig = plt.figure(figsize=(10,2))
            for each in range(num*5, 5*(num+1)):
                plt.subplot(1, 5, each+1-(num*5))
                plt.imshow(self.test_plot_incorrect[each].permute(1,2,0), cmap='bone')
                Title = f"actual label:{self.test_plot_arr_incorrect[each]['actual']}\n"                     f"predicted label:{self.test_plot_arr_incorrect[each]['predicted']}"
                plt.title(Title, color='orangered')
            plt.show()
        


# ### Initializing the MLP model

# In[24]:


model = MLP_PyTorch(num_input_nodes=28*28, num_hidden_nodes_1=256, num_hidden_nodes_2=256, num_output_nodes=10)


# ### Training the model

# In[25]:


model.fit(train_loader, val_loader, learning_rate=0.005)


# ### Testing and performance

# In[26]:


model.test(test_loader)


# ---
# # Question-2 (Artificial Neural Network from scratch)
# ---

# ## Preprocesing and Data visualization

# ### Loading the dataset

# In[27]:


abalone_df = pd.read_csv('abalone.csv')
abalone_df_raw = abalone_df.copy()


# In[28]:


abalone_df


# ### Data Analysis

# In[29]:


abalone_df.info()


# In[30]:


abalone_df.describe()


# In[31]:


Classes = np.array(sorted(abalone_df.Rings.unique()))
Classes


# In[32]:


features = abalone_df.columns.tolist()[:-1]
features


# ### Data visualization

# In[33]:


abalone_df.hist(figsize=(10,10))
plt.show()


# ### Encoding categorical features

# In[34]:


abalone_df['Sex'] = abalone_df['Sex'].astype('category').cat.codes


# ### Binning the labels to convert into classification problem

# In[35]:


Classes


# In[36]:


labels_array = list(np.array(abalone_df['Rings']))
# labels_array


# In[37]:


for each in range(len(labels_array)):
    if labels_array[each] <= 10:
        labels_array[each] = 'young-age'
    elif labels_array[each] <= 20:
        labels_array[each] = 'middle-age'
    else:
        labels_array[each] = 'old-age'


# In[38]:


abalone_df.Rings = labels_array


# In[39]:


abalone_df


# In[40]:


abalone_df['Rings'][abalone_df['Rings']=='young-age'] = 0
abalone_df['Rings'][abalone_df['Rings']=='middle-age'] = 1
abalone_df['Rings'][abalone_df['Rings']=='old-age'] = 2


# In[41]:


abalone_df


# In[42]:


abalone_X = abalone_df[features].copy()
abalone_y = abalone_df['Rings'].copy()


# In[104]:


figure = plt.figure(figsize=(10,7))
sns.heatmap(abalone_df.corr(), annot=True)
plt.show()


# ### Stratified Splitting into train-validation-test sets

# In[43]:


def stratified_splitting(dataframe, feature_col_names, label_col_name, train=0.7, val=0.1, test=0.2):
    data = np.array(dataframe)
    
    all_cols = feature_col_names.copy()
    all_cols.append(label_col_name)
    
    train_data = []
    val_data = []
    test_data = []
    
    class_array = data[:, -1]
    
    unique_classes = np.unique(class_array)
    
    for each in unique_classes:
        datapoints = data[class_array==each]
        np.random.shuffle(datapoints)
        num_datapoints = len(datapoints)
        num_train = int(train*num_datapoints)
        num_val = int(val*num_datapoints)
        
        train_data.append(datapoints[:num_train])
        val_data.append(datapoints[num_train: num_train+num_val])
        test_data.append(datapoints[(num_train+num_val):])

    train_data = np.concatenate(train_data)
    val_data = np.concatenate(val_data)
    test_data = np.concatenate(test_data)
    
    train_data = pd.DataFrame(train_data)
    val_data = pd.DataFrame(val_data)
    test_data = pd.DataFrame(test_data)
    
    train_data.columns = all_cols
    val_data.columns = all_cols
    test_data.columns = all_cols
    
    train_X = train_data[feature_col_names]
    train_y = train_data[label_col_name]
    
    val_X = val_data[feature_col_names]
    val_y = val_data[label_col_name]
    
    test_X = test_data[feature_col_names]
    test_y = test_data[label_col_name]
    
    return (train_X, train_y, val_X, val_y, test_X, test_y)
    


# In[44]:


train_X, train_y, val_X, val_y, test_X, test_y =     stratified_splitting(dataframe=abalone_df, feature_col_names=features, label_col_name='Rings')


# In[45]:


train_X


# In[46]:


val_X


# In[47]:


test_X


# In[48]:


# train_X, val_test_X, train_y, val_test_y = train_test_split(abalone_X, abalone_y, test_size=0.3, random_state=0, stratify=abalone_y)
# test_X, val_X, test_y, val_y = train_test_split(val_test_X, val_test_y, test_size=0.33, random_state=0, stratify=val_test_y)


# ## Multi-layer perceptron from scratch

# In[49]:


class MLP:
    def __init__(self, X, y, hidden_layers=None, num_hidden_layers=2, num_hidden_nodes=None,                  activation_function='sigmoid', lr=0.05, weights_type='Random', iterations=100):
        '''
        Inputs: training dataset and labels, information regarding hidden layers,
                activation function, learning rate, weights type for initialization, number of iterations
        
        Constructor for the Multi-layer Perceptron
        '''
        self.X = X
        self.y = y
        
        self.num_datapoints, self.num_input_nodes = np.array(self.X).shape
        self.output_nodes = np.sort(np.unique(self.y))
        self.num_output_nodes = len(self.output_nodes)
            
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
            self.num_hidden_layers = len(hidden_layers)
        else:
            self.num_hidden_layers = num_hidden_layers
            if num_hidden_nodes is None:
                self.num_hidden_nodes = int(math.sqrt(self.num_input_nodes*self.num_output_nodes))
            else:
                self.num_hidden_nodes = num_hidden_nodes
                
            self.hidden_layers = [self.num_hidden_nodes for each in range(self.num_hidden_layers)]
        
        self.activation_function = activation_function
        
        # initializing weights
        self.weights = list()
        if weights_type == 'Random':
            for each in range(self.num_hidden_layers+1):
                if each == 0:
                    self.weights.append(np.random.rand(self.hidden_layers[each], self.num_input_nodes))
                elif each == self.num_hidden_layers:
                    self.weights.append(np.random.rand(self.num_output_nodes, self.hidden_layers[each-1]))
                else:
                    self.weights.append(np.random.rand(self.hidden_layers[each], self.hidden_layers[each-1]))

        elif weights_type == 'Zero':
            for each in range(self.num_hidden_layers+1):
                if each == 0:
                    self.weights.append(np.zeros((self.hidden_layers[each], self.num_input_nodes)))
                elif each == self.num_hidden_layers:
                    self.weights.append(np.zeros((self.num_output_nodes, self.hidden_layers[each-1])))
                else:
                    self.weights.append(np.zeros((self.hidden_layers[each], self.hidden_layers[each-1])))

        elif weights_type == 'Constant':
            for each in range(self.num_hidden_layers+1):
                if each == 0:
                    self.weights.append(np.ones((self.hidden_layers[each], self.num_input_nodes)))
                elif each == self.num_hidden_layers:
                    self.weights.append(np.ones((self.num_output_nodes, self.hidden_layers[each-1])))
                else:
                    self.weights.append(np.ones((self.hidden_layers[each], self.hidden_layers[each-1])))
            
        # initializing biases
        self.biases = list()
        for each in range(len(self.weights)):
            self.biases.append(np.ones((self.weights[each].shape[0], 1)))
        
        '''
        print('Weights shapes:-')
        print([each.shape for each in self.weights])
        
        print('Biases shapes:-')
        print([each.shape for each in self.biases])
        '''
        
        # layers before activation
        self.in_layers = list()
        # layers after activation
        self.out_layers = list()
        
        # learning rate
        self.lr = lr
        
        # number of iterations
        self.iterations = iterations
        self.current_itr = 1
        
        self.f_counter = 0
        self.b_counter = 0
        
        self.gradients_out_layers = list()
        self.gradients_weights = list()
        self.gradients_biases = list()
        
        # expected outputs matrix
        self.expected_outputs = np.zeros((self.num_output_nodes, self.num_datapoints))
        for each in range(len(self.y)):
            self.expected_outputs[:, each][self.y[each]] = 1
            
        self.ActivationFunctions = {'sigmoid': self.sigmoid, 'tanh': self.tanh, 'relu': self.relu}
        self.Derivatives = {'sigmoid': self.deriv_sigmoid, 'tanh': self.deriv_tanh, 'relu': self.deriv_relu}
        
        self.iterations_array = list()
        self.costs_array = list()
        self.train_acc_array = list()
        self.val_acc_array = list()
        
    
    '''activation functions for the hidden layers'''
    # Sigmoid
    def sigmoid(self, x):
        '''
        Sigmoid function
        Returns the output of sigmoid function applied on the input
        '''
        x = x.astype(np.float64)
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self, x):
        '''
        Returns the derivative of sigmoid function for the input
        '''
        x = x.astype(np.float64)
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    # tanH
    def tanh(self, x):
        '''
        tanH function
        Returns the output of hyperbolic tangent function applied on the input
        '''
        x = x.astype(np.float64)
        return np.tanh(x)

    def deriv_tanh(self, x):
        '''
        Returns the derivative of tanh function for the input
        '''
        x = x.astype(np.float64)
        return 1 - np.tanh(x)**2


    # ReLU
    def relu(self, x):
        '''
        ReLU function
        Returns the output of rectified linear unit function applied on the input
        '''
        x = x.astype(np.float64)
        return np.maximum(0, x)

    def deriv_relu(self, x):
        '''
        Returns the derivative of the relu function for the input
        '''
        return np.array([(each >= 0) for each in x])

    
    '''activation function for the output layer'''
    # Softmax function
    def softmax(self, x):
        '''
        Softmax function
        Returns the output of softmax function applied on the input
        '''
        x = x.astype(np.float64)
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    
    # Cost function
    def categorical_cross_entropy_loss(self, expected, predicted):
        '''
        Categorical Cross Entropy loss function
        '''
        loss = -np.sum(expected*(np.log(predicted+1e-15))) / expected.shape[0]
        return loss

    
    def forward_propagation(self):
        '''
        Forward Propagation of the input
        '''
        activation_function = self.ActivationFunctions[self.activation_function]
        
        if self.f_counter == 0:
            self.f_counter = 1
            
            input_layer = np.array(self.X).T
            self.in_layers.append(input_layer)
            self.out_layers.append(input_layer)
            
            for each in range(self.num_hidden_layers+1):
                prev_layer = self.in_layers[each]
                weight = self.weights[each]
                
                new_layer = np.dot(weight, prev_layer) + self.biases[each]
                self.in_layers.append(new_layer)
                
                if each != self.num_hidden_layers:
                    self.out_layers.append(activation_function(new_layer))
                else:
                    self.out_layers.append(self.softmax(new_layer))
                    
        else:
            num_layers = len(self.in_layers)
            for each in range(1, num_layers):
                prev_layer = self.in_layers[each-1]
                weight = self.weights[each-1]
                
                new_layer = np.dot(weight, prev_layer) + self.biases[each-1]
                self.in_layers[each] = new_layer
                
                if each != num_layers-1:
                    self.out_layers[each] = activation_function(new_layer)
                else:
                    self.out_layers[each] = self.softmax(new_layer)
        '''
        print('Layers:-')
        print(f'in_layers: {[each.shape for each in self.in_layers]}')
        print(f'out_layers: {[each.shape for each in self.out_layers]}')
        '''

        
    def backward_propagation(self):
        '''
        Backward Propagation of the error using stochastic gradient descent
        Also updates the weights
        '''
        derivative = self.Derivatives[self.activation_function]
        
        num_layers = len(self.out_layers)
        
        if self.b_counter == 0:
            self.b_counter = 1
            
            # for the output layer (softmax is used here)
            self.gradients_out_layers.append(self.out_layers[num_layers-1] - self.expected_outputs)
            self.gradients_weights.append(1/self.num_datapoints *                     np.dot(self.gradients_out_layers[0], self.out_layers[num_layers-2].T))
            self.gradients_biases.append(1/self.num_datapoints *                     np.sum(self.gradients_out_layers[0], axis=1, keepdims=True))
            
            # for the rest of the layers:
            for each in range(num_layers-2, 0, -1):
                self.gradients_out_layers.append(                    np.dot(self.weights[each].T, self.gradients_out_layers[num_layers-2-each])                        * derivative(self.in_layers[each]))
                self.gradients_weights.append(1/self.num_datapoints *                     np.dot(self.gradients_out_layers[num_layers-1-each], self.out_layers[each-1].T))
                self.gradients_biases.append(1/self.num_datapoints *                     np.sum(self.gradients_out_layers[num_layers-1-each], axis=1, keepdims=True))
                
        else:
            # for the output layer (softmax is used here)
            self.gradients_out_layers[0] = self.out_layers[num_layers-1] - self.expected_outputs
            self.gradients_weights[0] = (1/self.num_datapoints *                     np.dot(self.gradients_out_layers[0], self.out_layers[num_layers-2].T))
            self.gradients_biases[0] = (1/self.num_datapoints *                     np.sum(self.gradients_out_layers[0], axis=1, keepdims=True))
            
            # for the rest of the layers:
            for each in range(num_layers-2, 0, -1):
                self.gradients_out_layers[num_layers-1-each] = (                    np.dot(self.weights[each].T, self.gradients_out_layers[num_layers-2-each])                        * derivative(self.in_layers[each]))
                self.gradients_weights[num_layers-1-each] = (1/self.num_datapoints *                     np.dot(self.gradients_out_layers[num_layers-1-each], self.out_layers[each-1].T))
                self.gradients_biases[num_layers-1-each] = (1/self.num_datapoints *                     np.sum(self.gradients_out_layers[num_layers-1-each], axis=1, keepdims=True))
        
        # updating weights
        for each in range(num_layers-1):
            self.weights[each] = self.weights[each] - self.lr*self.gradients_weights[num_layers-2-each]
            self.biases[each] = self.biases[each] - self.lr*self.gradients_biases[num_layers-2-each]
            
    
    def return_training_accuracy(self):
        '''
        Returns training accuracy
        '''
        predicted_y = self.out_layers[-1].T
        prediction_array = list()
        
        for each in range(self.num_datapoints):
            prediction_array.append(np.argmax(predicted_y[each]))
        prediction_array = np.array(prediction_array)
        
        # return prediction_array
        accuracy = 0
        for each in range(self.num_datapoints):
            if self.y[each] == prediction_array[each]:
                accuracy += 1
        accuracy /= self.num_datapoints
        
        return accuracy
    
    
    def return_training_cost(self):
        '''
        Returns training cost
        '''
        return self.categorical_cross_entropy_loss(self.expected_outputs.T, self.out_layers[-1].T)
    
    
    def start(self, val_X, val_y, print_cost=True):
        '''
        Starts training the model, updates the weights and also prints cost value for each 100th iteration
        '''
        for each in range(1, self.iterations+1):
            self.forward_propagation()
            self.backward_propagation()

            if each % 100 == 0:
                cost = self.return_training_cost()
                self.iterations_array.append(each)
                self.costs_array.append(cost)
                self.train_acc_array.append(self.return_training_accuracy())
                self.val_acc_array.append(self.score(val_X, val_y))
                
                if print_cost:
                    # print(f'iteration = {each+1}, training accuracy = {self.return_training_accuracy()}')
                    print(f'iteration = {each}, cost = {cost}')
            
        
    def score(self, X, y):
        '''
        Input: testing dataset and labels
        
        Returns accuracy score for the predictions on the given data
        '''
        activation_function = self.ActivationFunctions[self.activation_function]
        
        in_layers = list()
        out_layers = list()
        
        input_layer = np.array(X).T
        in_layers.append(input_layer)
        out_layers.append(input_layer)

        for each in range(self.num_hidden_layers+1):
            prev_layer = in_layers[each]
            weight = self.weights[each]

            new_layer = np.dot(weight, prev_layer) + self.biases[each]
            in_layers.append(new_layer)

            if each != self.num_hidden_layers:
                out_layers.append(activation_function(new_layer))
            else:
                out_layers.append(self.softmax(new_layer))
                
        num_datapoints = len(y)
        predicted_y = out_layers[-1].T
        prediction_array = list()
        
        for each in range(num_datapoints):
            prediction_array.append(np.argmax(predicted_y[each]))
        prediction_array = np.array(prediction_array)
        
        # return prediction_array
        accuracy = 0
        for each in range(num_datapoints):
            if y[each] == prediction_array[each]:
                accuracy += 1
        accuracy /= num_datapoints
        
        return accuracy

    def plots(self):
        fig = plt.figure(figsize=(5, 12))
        plt.subplot(3, 1, 1)
        plt.plot(self.iterations_array, self.costs_array, 'o-')
        plt.xlabel('number of iterations')
        plt.ylabel('cost values')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.iterations_array, self.train_acc_array, 'o-')
        plt.xlabel('number of iterations')
        plt.ylabel('train accuracy values')
        
        plt.subplot(3, 1, 3)
        plt.plot(self.iterations_array, self.val_acc_array, 'o-')
        plt.xlabel('number of iterations')
        plt.ylabel('validation accuracy values')
        
        plt.show()


# ---
# ## Sigmoid activation function
# ---

# ### Random weight initialization

# In[50]:


mlp_sigmoid_random = MLP(train_X, train_y, hidden_layers=[4], weights_type='Random', lr=0.3, iterations=3000)
mlp_sigmoid_random.start(val_X, val_y)


# In[51]:


mlp_sigmoid_random.plots()


# In[52]:


mlp_sigmoid_random.score(test_X, test_y)


# ### Constant weight initialization

# In[53]:


mlp_sigmoid_constant =     mlp_random = MLP(train_X, train_y, hidden_layers=[4], weights_type='Constant', lr=0.4, iterations=3000)
mlp_sigmoid_constant.start(val_X, val_y)


# In[54]:


mlp_sigmoid_constant.plots()


# In[55]:


mlp_sigmoid_constant.score(test_X, test_y)


# ### Zero weight initialization

# In[56]:


mlp_sigmoid_zero =     mlp_random = MLP(train_X, train_y, hidden_layers=[4], weights_type='Zero', lr=0.3, iterations=3000)
mlp_sigmoid_zero.start(val_X, val_y)


# In[57]:


mlp_sigmoid_zero.plots()


# In[58]:


mlp_sigmoid_zero.score(test_X, test_y)


# ---
# ## ReLU activation function
# ---

# ### Random weight initialization

# In[60]:


mlp_relu_random = MLP(train_X, train_y, hidden_layers=[4], weights_type='Random',                 activation_function='relu', lr=0.27, iterations=3000)
mlp_relu_random.start(val_X, val_y)


# In[61]:


mlp_relu_random.plots()


# In[62]:


mlp_relu_random.score(test_X, test_y)


# ### Constant weights initialization

# In[63]:


mlp_relu_constant = MLP(train_X, train_y, hidden_layers=[4], weights_type='Constant',                 activation_function='relu', lr=0.5, iterations=3000)
mlp_relu_constant.start(val_X, val_y)


# In[64]:


mlp_relu_constant.plots()


# In[65]:


mlp_relu_constant.score(test_X, test_y)


# ### Zero weights initialization

# In[66]:


mlp_relu_zero = MLP(train_X, train_y, hidden_layers=[4], weights_type='Zero',                 activation_function='relu', lr=0.225, iterations=3000)
mlp_relu_zero.start(val_X, val_y)


# In[67]:


mlp_relu_zero.plots()


# In[68]:


mlp_relu_zero.score(test_X, test_y)


# ---
# ## tanH activation function
# ---

# ### Random weights initialialization

# In[69]:


mlp_tanh_random = MLP(train_X, train_y, hidden_layers=[4], weights_type='Random',                 activation_function='tanh', lr=0.35, iterations=3000)
mlp_tanh_random.start(val_X, val_y)


# In[70]:


mlp_tanh_random.plots()


# In[71]:


mlp_tanh_random.score(test_X, test_y)


# ### Constant weights initialization

# In[72]:


mlp_tanh_constant = MLP(train_X, train_y, hidden_layers=[4], weights_type='Constant',                 activation_function='tanh', lr=0.2, iterations=3000)
mlp_tanh_constant.start(val_X, val_y)


# In[73]:


mlp_tanh_constant.plots()


# In[74]:


mlp_tanh_constant.score(test_X, test_y)


# ### Zero weights initialization

# In[75]:


mlp_tanh_zero = MLP(train_X, train_y, hidden_layers=[4], weights_type='Zero',                 activation_function='tanh', lr=0.7, iterations=3000)
mlp_tanh_zero.start(val_X, val_y)


# In[76]:


mlp_tanh_zero.plots()


# In[77]:


mlp_tanh_zero.score(test_X, test_y)


# In[78]:


def plot_comparision(plot_dict, xlab, ylab, Title):
    names = list(plot_dict.keys())
    values = list(plot_dict.values())
    
    plt.bar(range(len(plot_dict)), values, tick_label=names)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(Title)
    plt.show()


# ## Comparision among different activation functions based on accuracy scores

# In[79]:


random_accuracies = [mlp_sigmoid_random.score(test_X, test_y),
                     mlp_relu_random.score(test_X, test_y),
                     mlp_tanh_random.score(test_X, test_y)]


# In[80]:


constant_accuracies = [mlp_sigmoid_constant.score(test_X, test_y),
                       mlp_relu_constant.score(test_X, test_y),
                       mlp_tanh_constant.score(test_X, test_y)]


# In[81]:


zero_accuracies = [mlp_sigmoid_zero.score(test_X, test_y),
                   mlp_relu_zero.score(test_X, test_y),
                   mlp_tanh_zero.score(test_X, test_y)]


# In[82]:


act_funcs = ['Sigmoid', 'ReLU', 'tanH']
act_range = [1, 2, 3]


# In[83]:


plt.plot(act_range, random_accuracies, 'o-', label='random weights initialization')
plt.plot(act_range, constant_accuracies, 'o-', label='constant weights initialization')
plt.plot(act_range, zero_accuracies, 'o-', label='zero weights initialization')
plt.xlabel('activation functions')
plt.ylabel('accuracy scores')
plt.xticks(act_range, act_funcs)
plt.legend()
plt.show()


# ## Comparision among different weight initializations

# In[84]:


sigmoid_accuracies = {'random': mlp_sigmoid_random.score(test_X, test_y),
                      'constant': mlp_sigmoid_constant.score(test_X, test_y),
                      'zero': mlp_sigmoid_zero.score(test_X, test_y)}


# In[85]:


relu_accuracies = {'random': mlp_relu_random.score(test_X, test_y),
                   'constant': mlp_relu_constant.score(test_X, test_y),
                   'zero': mlp_relu_zero.score(test_X, test_y)}


# In[86]:


tanh_accuracies = {'random': mlp_tanh_random.score(test_X, test_y),
                   'constant': mlp_tanh_constant.score(test_X, test_y),
                   'zero': mlp_tanh_zero.score(test_X, test_y)}


# In[87]:


plot_comparision(sigmoid_accuracies, 'weight initializations', 'accuracy score', 'Sigmoid')


# In[88]:


plot_comparision(relu_accuracies, 'weight initializations', 'accuracy score', 'ReLU')


# In[89]:


plot_comparision(tanh_accuracies, 'weight initializations', 'accuracy score', 'tanH')


# ## Changing number of hidden nodes

# In[90]:


def change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function,                        start=2, stop=20, step=1, weights_type='Random', lr=0.2, iterations=3000):

    accuracy_scores_arr = []
    hidden_nodes_arr = []
    costs_arr_arr = []
    iterations_arr_arr = []
    val_acc_arr_arr = []
    
    for each in range(start, stop+1, step):
        mlp = MLP(train_X, train_y, hidden_layers=[each],              activation_function=activation_function, weights_type=weights_type, lr=lr, iterations=iterations)
        mlp.start(val_X, val_y, print_cost=False)
        
        accuracy_scores_arr.append(mlp.score(test_X, test_y))
        hidden_nodes_arr.append(each)
        costs_arr_arr.append(mlp.costs_array)
        iterations_arr_arr.append(mlp.iterations_array)
        val_acc_arr_arr.append(mlp.val_acc_array)
    
    print(f'number of hidden nodes: {hidden_nodes_arr}')
    print(f'corresponding accuracy: {accuracy_scores_arr}')
    
    nums = len(hidden_nodes_arr)
    
    
    '''plotting'''
    fig = plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for each in range(nums):
        plt.plot(iterations_arr_arr[each], costs_arr_arr[each], 'o-', label=hidden_nodes_arr[each])
    plt.xlabel('number of iterations')
    plt.ylabel('cost values')
    # plt.title('cost values vs number of iterations')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for each in range(nums):
        plt.plot(iterations_arr_arr[each], val_acc_arr_arr[each], 'o-', label=hidden_nodes_arr[each])
    plt.xlabel('number of iterations')
    plt.ylabel('validation accuracy scores')
    # plt.title('validation accuracy scores vs number of iterations')
    plt.legend()
    
    plt.show()
    
    plt.plot(hidden_nodes_arr, accuracy_scores_arr, 'o-')
    plt.xlabel('number of hidden nodes')
    plt.ylabel('test accuracy scores')
    # plt.title('accuracy scores vs number of hidden nodes')
    plt.show()
    


# ---
# ### for sigmoid activation function
# ---

# #### random weights initialization 

# In[91]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='sigmoid',                        start=1, stop=16, step=3, weights_type='Random', lr=0.25, iterations=2000)


# #### constant weights initialization

# In[92]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='sigmoid',                        start=1, stop=21, step=4, weights_type='Constant', lr=0.05, iterations=2500)


# #### zero weights initialization

# In[93]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='sigmoid',                        start=1, stop=16, step=3, weights_type='Zero', lr=0.25, iterations=2500)


# ---
# ### for relu activation function
# ---

# #### random weights initialization

# In[94]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='relu',                        start=1, stop=16, step=3, weights_type='Random', lr=0.1, iterations=2500)


# #### constant weights initialization

# In[95]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='relu',                        start=1, stop=16, step=3, weights_type='Constant', lr=0.25, iterations=2500)


# #### zero weights initialization

# In[96]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='relu',                        start=1, stop=21, step=5, weights_type='Zero', lr=0.2, iterations=2500)


# ---
# ### for tanh activation function
# ---

# #### random weights initialization

# In[97]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='tanh',                        start=1, stop=16, step=3, weights_type='Random', lr=0.25, iterations=2500)


# #### constant weights initialization

# In[98]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='tanh',                        start=1, stop=16, step=3, weights_type='Constant', lr=0.05, iterations=2500)


# #### zero weights initialization

# In[99]:


change_hidden_nodes(train_X, train_y, val_X, val_y, test_X, test_y, activation_function='tanh',                        start=1, stop=16, step=3, weights_type='Zero', lr=0.25, iterations=2500)


# ## Saving and loading MLP weights (using Pickle)

# ### Saving the weights

# In[100]:


save_name = 'model_weights.pkl'
with open(save_name, 'wb') as file:  
    pickle.dump(mlp_sigmoid_random.weights, file)


# ### Loading the weights

# In[101]:


with open(save_name, 'rb') as file:  
    pickled_weights = pickle.load(file)

pickled_weights

