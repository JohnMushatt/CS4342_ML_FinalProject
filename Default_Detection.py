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
from sklearn.metrics import confusion_matrix, classification_report, auc, precision_recall_curve, roc_curve
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

X_train = shuffle(X_train)
X_test = shuffle(X_test)
X_validation