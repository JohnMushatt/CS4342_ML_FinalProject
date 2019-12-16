from collections import Counter
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, auc, precision_recall_curve, roc_curve, \
    plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
from tensorboard import notebook

# %matplotlib notebook

data = pd.read_csv('trainFt.csv')
labels = pd.read_csv('trainLb.csv')
label_data = labels.iloc[:,0]
data.head()
print(data.columns)
data.insert(23,'Class',label_data,True)
print(data.columns)

defaulting = shuffle(data[data.Class==1])
non_defaulting = shuffle(data[data.Class==0])
# Produce a training set of 80% of fraudulent and 80% normal transactions
X_train = defaulting.sample(frac=0.8)
X_train = pd.concat([X_train, non_defaulting.sample(frac = 0.8)], axis = 0)
# Split remainder into testing and validation
remainder = data.loc[~data.index.isin(X_train.index)]
X_test = remainder.sample(frac=0.7)
X_validation = remainder.loc[~remainder.index.isin(X_test.index)]


# Reshuffle stuff to make sure random


# Shuffle the datasets once more to ensure random feeding into the classification algorithms
X_train = shuffle(X_train)
X_test = shuffle(X_test)
X_validation = shuffle(X_validation)
X_train_ = shuffle(X_train)
X_test_ = shuffle(X_test)
X_validation_ = shuffle(X_validation)
data_resampled = pd.concat([X_train_, X_test_, X_validation_])

# Normalize the data to an average of 0 and std of 1, but do not touch the Class column
for feature in X_train.columns.values[:-1]:
    mean, std = data[feature].mean(), data[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std
    X_validation.loc[:, feature] = (X_validation[feature] - mean) / std
for feature in X_train_.columns.values[:-1]:
    mean, std = data_resampled[feature].mean(), data_resampled[feature].std()
    X_train_.loc[:, feature] = (X_train_[feature] - mean) / std
    X_test_.loc[:, feature] = (X_test_[feature] - mean) / std
    X_validation_.loc[:, feature] = (X_validation_[feature] - mean) / std


# Create labels
y_train = X_train.Class
y_test = X_test.Class
y_validation = X_validation.Class
y_train_ = X_train_.Class
y_test_ = X_test_.Class
y_validation_ = X_validation_.Class
# Remove labels from X's
X_train = X_train.drop(['Class'], axis=1)
X_train_ = X_train_.drop(['Class'], axis=1)
X_test = X_test.drop(['Class'], axis=1)
X_test_ = X_test_.drop(['Class'], axis=1)
X_validation = X_validation.drop(['Class'], axis=1)
X_validation_ = X_validation_.drop(['Class'], axis=1)

# Pickle and save the dataset
dataset = {'X_train' : X_train,
           'X_train_': X_train_,
           'X_test': X_test,
           'X_test_': X_test,
           'X_validation': X_validation,
           'X_validation_': X_validation_,
           'y_train': y_train,
           'y_train_': y_train_,
           'y_test': y_test,
           'y_test_': y_test_,
           'y_validation': y_validation,
           'y_validation_': y_validation_}
#with open('pickle/data_with_resample_apr19.pkl', 'wb+') as f:
 #   pickle.dump(dataset, f)

# Random Forest on unsampled training data
# rf_n_est_100_unsampled_unweighted
rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
rf.fit(dataset['X_train'], dataset['y_train'])
y_pred = rf.predict(dataset['X_test'])
plot_confusion_matrix(dataset['y_test'], y_pred)