'''
	File name: dnn.py
	Program Description: A program that creates, compiles and evaluates a deep neural network model.
	Date: 22 July 2018
	Author: Loria Roie Grace N. Malingan 
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Reshape, Flatten
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
import pandas as pd
#import matplotlib as plt
import numpy as np
from time import time

# reading csv file containing drone image data using pandas
filename = '_output.csv'
data = pd.read_csv(filename)
print(data.head())

# reshaping
table = data.pivot_table(index='Plot', values=['R.x','G.x','B.x','R.y','G.y','B.y','HT.x'])
print(table.head())
#print data.dtypes
#print data.columns

# identifying features and target labels / isolating target label from the feature data
y = table.get('HT.x')
X = table.drop('HT.x', axis=1)

# splitting the dataset into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print X_test.shape

#--- Deep Neural Network  ---#
# Create model
def build_model():
	# Initialize model
	model = Sequential()
	# Add input layer	
	model.add((Dense(64, input_dim=6)))
	model.add(Reshape((64,1)))
	# Add convolutional layer
	model.add(Conv1D(8, kernel_size=(3), strides=(1), activation='relu'))
	# Add Max-Pooling
	model.add(MaxPooling1D(pool_size=(3), strides=(1)))
	# Add Dropout Layer
	model.add(Dropout(rate=0.2))
	# Add first fully-connected layer
	model.add(Dense(32, activation='relu'))
	# Add Dropout Layer
	model.add(Dropout(rate=0.1))
	# Add second fully-connected layer
	model.add(Dense(1, activation='relu'))
	# Add Dropout Layer
	model.add(Dropout(rate=0.5))
	model.add(Flatten())
	# Add output layer
	model.add(Dense(1))
	return model
'''
# Uncomment to view map
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,5])
'''

model = build_model()
print(model.summary()) # display model summary
# visualize model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # Compile model

# Callbacks
tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_images=True)
early_stop = EarlyStopping(monitor='loss', patience=20)

# Train the model
history = model.fit(X_train, y_train, epochs=6000, verbose=1, callbacks=[tensorboard, early_stop])
# plot_history(history)

# Use model to predict with the test data
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

# Evaluate model
mse_value, mae_value = model.evaluate(X_test, y_test, verbose=0)
print("MSE: " + str((mse_value)))
print("MAE: " + str((mae_value)))

print("Prediction score: " + str((score)))

# end