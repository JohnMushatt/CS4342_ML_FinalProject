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



testing_set = testing_data
#print(testing_set.shape)

# Get just the features of the training set
training_set_features = training_set.iloc[:, :-1]
training_set_labels = training_set.iloc[:, -1]
#print(training_set_features.shape)

# Validation data
validation_set_features = validation_set.iloc[:, :-1]
validation_set_labels = validation_set.iloc[:, -1]
#print(validation_set_features.shape)

sc = StandardScaler()

# Fit on training set only.
sc.fit(training_set_features)

# Apply transform to both the training set and the test set.
training_set_transformed = sc.transform(training_set_features)
validation_set_transformed = sc.transform(validation_set_features)
testing_set = sc.transform(testing_set)

# Get PCA data
pca = PCA(.95)
pca.fit(training_set_transformed)

# Apply the PCA result to the training and validation
X_train = pca.transform(training_set_transformed)
X_test = pca.transform(validation_set_transformed)
principal_testing_data = pca.transform(testing_set)

print(X_train.shape)
print(X_test.shape)
print(principal_testing_data.shape)


#Build the autoencoder
input_dim = X_train.shape[1]
print(input_dim)

encoding_dim = 14
input_layer = Input(shape=(input_dim,))
nb_epoch = 10  # TODO CHANGE BACK TO 100
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
#Print some of the stats
print(training_set_features.shape)
print(training_set_labels.shape)
print(validation_set_features.shape)
print(validation_set_labels.shape)
autoencoder.summary()

history = autoencoder.fit(X_train, X_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test,  X_test),
                          verbose=1,
                          ).history
#autoencoder = load_model('model.h5')
#autoencoder.summary()
# loss = history['loss']
# val_loss = history['val_loss']
# epochs = range(nb_epoch)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

#Predict it
pred =autoencoder.predict(principal_testing_data)
