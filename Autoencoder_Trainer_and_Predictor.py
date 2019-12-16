import csv

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# importing or loading the dataset
training_features = pd.read_csv('trainFt.csv')
training_lables = pd.read_csv('trainLb.csv')
testing_data = pd.read_csv('testFt.csv')

# Data with labels attached
training_data = training_features.join(training_lables)
# Split the data into 80% training 20% validation
training_set, validation_set = train_test_split(training_data, test_size=.2)

testing_set_transformed = testing_data
# print(testing_set.shape)
training_set = training_set[training_set.Class==0]
#validation_set = validation_set[validation_set.Class==0]
# Get just the features of the training set
training_set_features = training_set.iloc[:, :-1]
training_set_labels = training_set.iloc[:, -1]
# print(training_set_features.shape)

# Validation data
validation_set_features = validation_set.iloc[:, :-1]
validation_set_labels = validation_set.iloc[:, -1]
# print(validation_set_features.shape)

sc = StandardScaler()

# Fit on training set only.
sc.fit(training_set_features)

# Apply transform to both the training set and the test set.
training_set_transformed = sc.transform(training_set_features)
validation_set_transformed = sc.transform(validation_set_features)
testing_set_transformed = sc.transform(testing_set_transformed)

# Get PCA data
pca = PCA(.95)
pca.fit(training_set_transformed)

# Apply the PCA result to the training and validation
X_train = pca.transform(training_set_transformed)
X_test = pca.transform(validation_set_transformed)
principal_testing_data = pca.transform(testing_set_transformed)

#X_train = training_features
#X_test = validation_set_features
print(X_train.shape)
print(X_test.shape)
print(principal_testing_data.shape)

# Build the autoencoder
input_dim = X_train.shape[1]
print(input_dim)

encoding_dim = 14
input_layer = Input(shape=(input_dim,))
nb_epoch = 100  # TODO CHANGE BACK TO 100
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
batch_size = 32
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='/media/old-tf-hackers-7/logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
# Print some of the stats
print(training_set_features.shape)
print(training_set_labels.shape)
print(validation_set_features.shape)
print(validation_set_labels.shape)
autoencoder.summary()

history = autoencoder.fit(X_train, X_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          ).history
print("Saving model")
autoencoder.save('model.h5', overwrite=True)
print("Loading model with acc .8386")
autoencoder = load_model('acc_8386.h5')
autoencoder.summary()
loss = history['loss']
val_loss = history['val_loss']
epochs = range(nb_epoch)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#plt.show()
plt.savefig('visuals/tr_val_loss.png', bbox_inches='tight')
# Predict it
pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - pred, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                         'true_class': validation_set_labels})
print(error_df.describe())
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class'] == 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)

#plt.show()
plt.savefig('visuals/normal_error_defaulting.png', bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
#plt.show()
plt.savefig('visuals/fraud_error.png', bbox_inches='tight')
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
plt.savefig('visuals/ROC.png', bbox_inches='tight')

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.show()
plt.savefig('visuals/Recall_v_precision.png', bbox_inches='tight')

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
#plt.show()
plt.savefig('visuals/precision_threshold.png', bbox_inches='tight')
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
#plt.show()
plt.savefig('visuals/recall_threshold.png', bbox_inches='tight')
threshold = 2.9

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
#plt.show()
plt.savefig('visuals/reconstruction_errror.png', bbox_inches='tight')

threshold = 1.33


#--------
#Predict the training testing data
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=['Normal', 'Defaulting'], yticklabels=['Normal', 'Defaulting'], annot=True,
            fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
#plt.show()
plt.savefig("visuals/confusion_matrix.png", bbox_inches='tight')


#Predict final output for project
final_test_predictions = autoencoder.predict(principal_testing_data)
final_mse = np.mean(np.power(principal_testing_data - final_test_predictions, 2), axis=1)
final_error_df = pd.DataFrame({'reconstruction_error': final_mse})
print(final_error_df.describe())
final_y_pred = [1 if e > threshold else 0 for e in final_error_df.reconstruction_error.values]

final_default_count =0
final_nondefault_count = 0
for e in final_y_pred:
    if(e==1):
        final_default_count = final_default_count+1
    else:
        final_nondefault_count = final_nondefault_count+1
training_default_count =0
training_nondefault_count = 0
for e in y_pred:
    if(e==1):
        training_default_count = training_default_count+1
    else:
        training_nondefault_count = training_nondefault_count+1

print("Default count: " + str(final_default_count) + " Non defaulting: " + str(final_nondefault_count))
print("Default percent of total final test sample " + str(final_default_count/(final_nondefault_count+final_default_count)))
print("default percent of total training sample " + str(training_default_count / (training_default_count + training_nondefault_count)))

with open('data/final_predictions.csv','w') as result_file:
    wr = csv.writer(result_file,dialect='excel')
    wr.writerow(final_y_pred)
result_file.close()