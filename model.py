# Import libraries
import numpy as np
import pandas as pd

# Load dataset
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Preprocessing the training set and test set
from preprocessing_data import processor_data

clean_data = processor_data(train_data, test_data)


# Find the best model in this problem
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = []

models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

names = []
results = []

for name, model in models:
    kfold = StratifiedKFold(
        n_splits=10,
        random_state=1,
        shuffle=True)
    
    cv_results = cross_val_score(
        model,
        clean_data['x_train'],
        clean_data['y_train'],
        cv = kfold,
        scoring='accuracy')
    
    names.append(name)
    results.append(cv_results)
    print(name, cv_results.mean(), cv_results.std())