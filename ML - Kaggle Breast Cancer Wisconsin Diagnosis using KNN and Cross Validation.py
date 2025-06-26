"""
ML | Kaggle Breast Cancer Wisconsin Diagnosis using KNN and Cross Validation
Last Updated : 22 May, 2024

Dataset : It is given by Kaggle from UCI Machine Learning Repository, in one of its challenges. It is a dataset of Breast Cancer patients with Malignant and Benign tumor. K-nearest neighbour algorithm is used to predict whether is patient is having cancer (Malignant tumour) or not (Benign tumour). Implementation of KNN algorithm for classification. Code : Importing Libraries
"""

# performing linear algebra
import numpy as np 

# data processing
import pandas as pd

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#Loading dataset
df = pd.read_csv("db/data.csv")

print (df.head)

#Data Info
df.info()

# We are dropping columns - 'id' and 'Unnamed: 32' as they have no role in prediction
df = df.drop(['Unnamed: 32', 'id'], axis = 1)
df = df.dropna()
print(df.shape)

# Converting the diagnosis value of M and B to a numerical value where M (Malignant) = 1 and B (Benign) = 0
def diagnosis_value(diagnosis):
    if diagnosis == 'M':
        return 1
    else:
        return 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

sns.lmplot(x = 'radius_mean', y = 'texture_mean', hue = 'diagnosis', data = df)
plt.show()

sns.lmplot(x ='smoothness_mean', y = 'compactness_mean', 
           data = df, hue = 'diagnosis')
plt.show()


# Input and Output data
X = np.array(df.iloc[:, 1:])
y = np.array(df['diagnosis'])


# Splitting data to training and testing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33, random_state = 42)

print(X_train)

# Using Sklearn
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)

# Prediction Score
predict = knn.score(X_test, y_test)

print(predict)
# 0.9627659574468085


# Performing Cross Validation 
neighbors = []
cv_scores = []

from sklearn.model_selection import cross_val_score
# perform 10 fold cross validation
for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(
        knn, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())



# : Misclassification error versus k
MSE = [1-x for x in cv_scores]

# determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is % d ' % optimal_k)

# plot misclassification error versus k
plt.figure(figsize = (10, 6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()


# The optimal number of neighbors is 13 