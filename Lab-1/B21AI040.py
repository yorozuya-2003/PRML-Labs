# -*- coding: utf-8 -*-
"""PRML_Lab-1_B21AI040.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1beWxBgjGwWpTTrvCIhvVxYktVypVQvJT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

"""# Problem-1

---

**Problem-1 (a)**



---
"""

file = pd.read_csv('/content/drive/MyDrive/PRML_Lab/Lab_1/demo.csv')

file

file.Data.tolist()

"""**Problem-1 (b)**

---
"""

user_input = input()
# user_input = 35
num_input = int(user_input)
print('Type of user input changed from', type(user_input), 'to', type(num_input))

"""**Problem-1 (c)**

---
"""

from datetime import datetime

demo_string = '11/18/03 12:00:00'
demo_string_to_datetime = datetime.strptime(demo_string, '%m/%d/%y %H:%M:%S')

print(type(demo_string), 'changed to', type(demo_string_to_datetime))
print(demo_string_to_datetime)

"""**Problem-1 (d)**

---
"""

# call external commands in python
import os
os.system("dir *.md")

"""**Problem-1 (e)**

---
"""

demo_list = [1, 2, 3, 2, 2, 4]
print('count of 2 in list:', demo_list.count(2))

"""**Problem-1 (f)**

---
"""

demo_list = [[1, 2], [1, [2, 3]], [1 , 3]]
flattened_list = []

def flatten(l, index):
  if(type(l[index]) != list):
    flattened_list.append(l[index])
  else:
    for i in range(len(l[index])):
      flatten(l[index], i)

for i in range(len(demo_list)):
  flatten(demo_list, i)
    
print('original list:', demo_list)
print('flattened list:', flattened_list)

"""**Problem-1 (g)**

---
"""

d1 = {'hello' : 'world', 'tanish' : 'pagaria'}
d2 = {1 : 2, 3 : 4}
d3 = {}
for key in d1:
  d3[key] = d1[key]

for key in d2:
  d3[key] = d2[key]

print(d3)

"""**Problem-1 (h)**

---
"""

demo_list = [1, 2, 3, 4, 4, 5, 6, 6, 7, 1, 2, 3]
new_list = []
for item in demo_list:
  if item not in new_list:
    new_list.append(item)

print(new_list)

"""**Problem-1 (i)**

---
"""

demo_dict = {'hello' : 'world', 'tanish' : 'pagaria'}
demo_key1 = 'tanish'
demo_key2 = 1

if demo_key1 in demo_dict:
  print('key-1 is present in dictionary.')
else:
  print('key-1 is not present in dictionary.')
if demo_key2 in demo_dict:
  print('key-2 is present in dictionary.')
else:
  print('key-2 is not present in dictionary.')

"""---



---

# Problem-2

---
"""

mat1 = np.array([[9, 2, 1], [4, 2, 2], [8, 2, 1]])
mat2 = np.array([[8, 6, 1], [9, 6, 9], [9, 4, 3]])

"""**Problem-2 (a)**

---
"""

print('first row of first matrix:', mat1[0])

"""**Problem-2 (b)**

---
"""

print('second column of second matrix:', mat2.T[1])

"""**Problem-2 (c)**

---
"""

print('matrix multiplication:-\n', mat1 @ mat2)

"""**Problem-2 (d)**

---
"""

print('element-wise multiplication:-\n', np.multiply(mat1, mat2))

"""**Problem-2 (e)**

---
"""

print('dot product between each column of first matrix and each column of second matrix:', np.sum(mat1 @ mat2))

"""---



---

# Problem-3

---
"""

df = pd.read_csv('/content/drive/MyDrive/PRML_Lab/Lab_1/Cars93.csv')

df

"""**Problem-3 (i)**

---
"""

# (a) model => ordinal scale
# (b) type => nominal scale
# (c) max. price => ratio scale
# (d) airbags => nominal scale

df.dtypes

def assign_scale(dataframe):
  ordinal = []
  nominal = []
  ratio = []
  interval = []

  col_list = df.columns.tolist()
  
  for col in col_list:
    if(df.dtypes[col] == 'object'):
      col_set = set()
      for each in df[col]:
        col_set.add(each)

      if len(col_set) < 10:
        nominal.append(col)
      else:
        ordinal.append(col)

    else:
      if df.dtypes[col] == int:
        interval.append(col)
      else:
        ratio.append(col)

  print('nominal:', nominal)
  print('ordinal:', ordinal)
  print('interval:', interval)
  print('ratio:', ratio)

assign_scale(df)

"""**Problem-3 (ii)**

---
"""

df = df.dropna(axis = 0)

df

"""**Problem-3 (iii)**

---
"""

col_list = df.columns.tolist()
num_col = len(col_list)

index_label_list = df.index.values.tolist()
# index_label_list

dtype_list = []
col_list = df.columns.tolist()
for col in col_list:
  if str(df[col][0]).isnumeric():
    dtype_list.append(int)
  else:
    dtype_list.append(type(df[col][0]))

# dtype_list

rows_to_be_removed = []
for row in range(len(df)):
  for col in range(num_col):
    if dtype_list[col] == str:
      if str(df.iloc[row, col]).isnumeric() and row not in rows_to_be_removed:
        rows_to_be_removed.append(row)

rows_to_be_removed

for r in range(len(rows_to_be_removed)):
    rows_to_be_removed[r] = index_label_list[rows_to_be_removed[r]]

rows_to_be_removed

df = df.drop(rows_to_be_removed)

df

"""**Problem-3 (iv)**

---
"""

df.dtypes

obj_list = []

for col in col_list:
  if df.dtypes[col] == 'object':
    obj_list.append(col)

obj_list

df[obj_list] = df[obj_list].astype('category')
for col in obj_list:
  df[col] = df[col].cat.codes
df

"""**Problem-3 (v)**

---
"""

df.dtypes

for col in col_list:
  if col not in obj_list:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

df

"""**Problem-3 (vi)**

---
"""

train, val, test = np.split(df.sample(frac=1), [int(0.7 * len(df)), int(0.9 * len(df))])

train

val

test

"""---



---

# Problem-4

---

**Problem-4 (a)**

---
"""

x = np.linspace(-10, 10, 1000)
y = 5*x + 4

plt.plot(x, y)
plt.show()

"""**Problem-4 (b)**

---
"""

x = np.linspace(10, 1000, 100)
y = np.log(x)

plt.plot(x, y)
plt.plot([10, 1000], np.log([10, 1000]), 'o')
plt.show()

"""**Problem-4 (c)**

---
"""

x = np.linspace(-10, 10, 100)
y = x**2

plt.plot(x, y)
plt.show()

"""**Problem-4 (d)**

---
"""

x = [0, 1, 2, 3, 4]
y = [2, 3, 4, 5, 6]

plt.plot(x, y)
plt.show()

"""# Problem-5

**Import the Necessary Python Libraries and Components**
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as f1s
from sklearn.metrics import accuracy_score as acc

"""**To Disable Convergence Warnings (For Custom Training)**"""

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

"""**1.) Input the Dataset**"""

# Dataset Reference :- https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

data = pd.read_csv('/content/drive/MyDrive/PRML_Lab/Lab_1/Wisconsin_DataSet.csv')
data

"""**2.) Convert the String Labels into easily-interpretable Numerics**"""

# Note :- There are many existing Encoders for converting String to Numeric Labels, but for convenience, we used Pandas.

condition_M = data.diagnosis == "M"
condition_B = data.diagnosis == "B"

data.loc[condition_M,"diagnosis"]=0
data.loc[condition_B,"diagnosis"]=1

data

"""**3.) Converting Dataframe into Numpy Arrays (Features and Labels)**"""

Y = data.diagnosis.to_numpy().astype('int')                                     # Labels

X_data = data.drop(columns=["id","diagnosis","Unnamed: 32"])
X = X_data.to_numpy()                                                           # Input Features

"""**4.) Splitting the Dataset into Train and Test Portions**"""

user_prompt = 0.3
user_enable = False

x_train,x_test,y_train,y_test = tts(X,Y,test_size=user_prompt,shuffle=user_enable)

"""**5.) Model Training and Predicting**"""

# Note :- Don't worry about the code snippet here, it is just to produce the predictions for the test data portion of each classifier

logistic_model = LR()
logistic_model.fit(x_train,y_train)
logistic_pred = logistic_model.predict(x_test)

decision_model = DTC()
decision_model.fit(x_train,y_train)
decision_pred = decision_model.predict(x_test)

"""**6.) Evaluation Metrics (Inbulit v/s Scaratch)**

## Confusion Matrix
"""

inbuilt_matrix_logistic = cm(y_test,logistic_pred)
inbuilt_matrix_decision = cm(y_test,decision_pred)

print("Confusion Matrix for Logistic Regression-based Predictions =>")
print(inbuilt_matrix_logistic)
print("Confusion Matrix for Decision Tree-based Predictions =>")
print(inbuilt_matrix_decision)

def confusion_matrix(y_test, y_pred):    
  true_pos = 0
  true_neg = 0
  false_pos = 0
  false_neg = 0

  for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 1:
      true_pos += 1
    elif y_test[i] == 0 and y_pred[i] == 0:
      true_neg += 1
    elif y_test[i] == 1 and y_pred[i] == 0:
      false_neg += 1
    elif y_test[i] == 0 and y_pred[i] == 1:
      false_pos += 1

  return np.array([[true_neg, false_pos], [false_neg, true_pos]])

"""## Average Accuracy"""

inbuilt_acc_logistic = acc(y_test,logistic_pred)
inbuilt_acc_decision = acc(y_test,decision_pred)

print("Accuracy for Logistic Regression-based Predictions =>",str(inbuilt_acc_logistic*100)+"%")
print("Accuracy for Decision Tree-based Predictions =>",str(inbuilt_acc_decision*100)+"%")

def avg_accuracy(y_test, y_pred):
  true_pos = 0
  true_neg = 0

  for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 1:
      true_pos += 1
    elif y_test[i] == 0 and y_pred[i] == 0:
      true_neg += 1

  return (true_pos + true_neg) / len(y_test)

"""## Precision"""

inbuilt_ps_logistic = ps(y_test,logistic_pred)
inbuilt_ps_decision = ps(y_test,decision_pred)

print("Precision for Logistic Regression-based Predictions =>",str(inbuilt_ps_logistic*100)+"%")
print("Precision for Decision Tree-based Predictions =>",str(inbuilt_ps_decision*100)+"%")

def precision(y_test, y_pred):
  true_pos = 0
  false_pos = 0

  for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 1:
      true_pos += 1
    elif y_test[i] == 0 and y_pred[i] == 1:
      false_pos += 1

  return true_pos / (true_pos + false_pos)

"""## Recall"""

inbuilt_rs_logistic = rs(y_test,logistic_pred)
inbuilt_rs_decision = rs(y_test,decision_pred)

print("Recall for Logistic Regression-based Predictions =>",str(inbuilt_rs_logistic*100)+"%")
print("Recall for Decision Tree-based Predictions =>",str(inbuilt_rs_decision*100)+"%")

def recall(y_test, y_pred):
  true_pos = 0
  false_neg = 0

  for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 1:
      true_pos += 1
    elif y_test[i] == 1 and y_pred[i] == 0:
      false_neg += 1

  return true_pos / (true_pos + false_neg)

"""## F-1 Score"""

inbuilt_f1s_logistic = f1s(y_test,logistic_pred)
inbuilt_f1s_decision = f1s(y_test,decision_pred)

print("F1-Score for Logistic Regression-based Predictions =>",str(inbuilt_f1s_logistic*100)+"%")
print("F1-Score for Decision Tree-based Predictions =>",str(inbuilt_f1s_decision*100)+"%")

def f1_score(y_test, y_pred):
  prec = precision(y_test, y_pred)
  rec = recall (y_test, y_pred)

  return 2 * prec * rec / (prec + rec)

"""## Class-Wise Accuracy"""

def class_accuracy(y_test, y_pred):
  true_pos = 0
  true_neg = 0
  false_pos = 0
  false_neg = 0

  for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 1:
      true_pos += 1
    elif y_test[i] == 0 and y_pred[i] == 0:
      true_neg += 1
    elif y_test[i] == 1 and y_pred[i] == 0:
      false_neg += 1
    elif y_test[i] == 0 and y_pred[i] == 1:
      false_pos += 1

  return 0.5 * (( true_pos / (true_pos + false_neg) + ( true_neg / (true_neg + false_pos) )))

"""## Sensitivity"""

def sensitivity(y_test, y_pred):
    return recall(y_test, y_pred)

"""## Specificity"""

def specificity(y_test, y_pred):
  true_neg = 0
  false_pos = 0

  for i in range(len(y_test)):
    if y_test[i] == 0 and y_pred[i] == 0:
      true_neg += 1
    elif y_test[i] == 0 and y_pred[i] == 1:
      false_pos += 1

  return true_neg / (true_neg + false_pos)