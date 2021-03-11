# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:43:01 2019
@author: mikhail.galkin
"""

## WORKSHOP 2019-03-20 KERAS
### NOTEBOOK: 01 Linear Regression
#---------------------------------------------------------------------------------------------------
import numpy as np

np.set_printoptions(suppress=True)

X = np.arange(-10, 11).reshape((21,1))
X
y = 2*X + 1
y

list(zip(X, y))
#---------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.plot(X, y, 'ro')
#display(plt.show())
#---------------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr
lr.fit(X, y)
#---------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error

y_pred = lr.predict(X)
y_pred
mse = mean_squared_error(y, y_pred)
mse
print(mse)
#Visualize Predictions
plt.plot(X, y_pred)
#display(plt.show())
#---------------------------------------------------------------------------------------------------
## 1. Define a N-Layer Network
import tensorflow as tf
tf.set_random_seed(42) # For reproducibility
np.random.seed(0)

from keras.models import Sequential
from keras.layers import Dense

# The Sequential model is a linear stack of layers.
model = Sequential()

model.add(Dense(units=1, input_dim=1, activation='linear'))

model.summary()
#---------------------------------------------------------------------------------------------------
## 2. Compile a Network
model.compile(loss='mse', optimizer='adam')
#---------------------------------------------------------------------------------------------------
## 3. Fit a Network
model.fit(X, y)
#---------------------------------------------------------------------------------------------------
keras_pred = model.predict(X)
keras_pred

def kerasPredPlot(keras_pred):
  plt.clf()
  plt.plot(X, y, 'ro', label='True')
  plt.plot(X, keras_pred, 'go', label='Keras Prediction')
  plt.legend(numpoints=1)
  display(plt.show())

kerasPredPlot(keras_pred)

history = model.fit(X, y, epochs=20, verbose = False) 

def viewModelLoss():
  plt.clf()
  plt.plot(history.history['loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  display(plt.show())

viewModelLoss()

history = model.fit(X, y, epochs=4000, verbose = False)
viewModelLoss()
#---------------------------------------------------------------------------------------------------
print(model.get_weights())
predicted_w = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]

print("predicted_w: ", predicted_w)
print("predicted_b: ", predicted_b)

history = model.fit(X, y, epochs=20000, verbose = False)
viewModelLoss()
#---------------------------------------------------------------------------------------------------
## 4. Evaluate Network
model.evaluate(X, y) # Prints loss value & metrics values for the model in test mode (both are MSE here)
#---------------------------------------------------------------------------------------------------
## 5. Make Predictions
keras_pred_1 = model.predict(X)
keras_pred_1
kerasPredPlot(keras_pred_1)
####################################################################################################



### NOTEBOOK: 02 KERAS
####################################################################################################
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(0)

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)
print(cal_housing.DESCR)
print(cal_housing.shape)
#---------------------------------------------------------------------------------------------------
import tensorflow as tf
tf.set_random_seed(42) # For reproducibility

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# Input layer
model.add(Dense(20, input_dim = 8, activation='relu')) 

# Automatically infers the input_dim based on the layer before it
model.add(Dense(20, activation='relu')) 

# Output layer
model.add(Dense(1, activation='linear')) 
#---------------------------------------------------------------------------------------------------
def build_model():
  return Sequential([Dense(20, input_dim=8, activation='relu'),
                    Dense(20, activation='relu'),
                    Dense(1, activation='linear')]) 
# Keep the last layer as linear because this is a regression problem
#---------------------------------------------------------------------------------------------------
model = build_model()
model.summary()


#---------------------------------------------------------------------------------------------------
from keras import metrics
from keras import losses

loss = "mse" # Or loss = losses.mse
metrics = ["mae", "mse"] # Or metrics = [metrics.mae, metrics.mse]

model.compile(optimizer="sgd", loss=loss, metrics=metrics)
model.fit(X_train, y_train, epochs=10)


#---------------------------------------------------------------------------------------------------

# Configure custom optimizer: 
from keras import optimizers

model = build_model()
optimizer=optimizers.Adam(lr=0.001)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(X_train, y_train, epochs=20)

#---------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def viewModelLoss():
  plt.clf()
  plt.plot(history.history['loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  display(plt.show())
viewModelLoss()

#---------------------------------------------------------------------------------------------------
model = build_model()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2)
viewModelLoss()


#---------------------------------------------------------------------------------------------------

" Advanced Keras "

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(0)

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)

print(cal_housing.DESCR)

#---------------------------------------------------------------------------------------------------
import pandas as pd

xTrainDF = pd.DataFrame(X_train, columns=cal_housing.feature_names)

print(xTrainDF.describe())

#---------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#---------------------------------------------------------------------------------------------------
import tensorflow as tf
tf.set_random_seed(42) # For reproducibility

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
  Dense(20, input_dim=8, activation='relu'),
  Dense(20, activation='relu'),
  Dense(1, activation='linear')
])

    
#---------------------------------------------------------------------------------------------------
model.compile(optimizer="adam", loss="rmse")
model.compile(optimizer="adam", loss="mse", metrics=["rmse"])

from keras import backend
 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
model.compile(optimizer="adam", loss="mse", metrics=["mse", rmse])


#---------------------------------------------------------------------------------------------------
history = model.fit(X_train, y_train, validation_split=.2, epochs=10, verbose=2)

import numpy as np
np.sqrt(history.history['mean_squared_error'][-1]) # Get MSE of last training epoch

model.compile(optimizer="adam", loss="mse", metrics=["mse"]) # Remove the RMSE metric




"  Convolutional Neural Networks  "

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import numpy as np
import os

vgg16Model = VGG16(weights='imagenet')

vgg16Model.summary()


######################################
def predict_images(images, model):
  for i in images:
    print ('processing image:', i)
    img = image.load_img(i, target_size=(224, 224))
    #convert to numpy array for Keras image formate processing
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability
    print('Predicted:', decode_predictions(preds, top=3)[0], '\n')


########################################
img_paths = ["/dbfs/mnt/training/dl/img/pug.jpg", 
             "/dbfs/mnt/training/dl/img/strawberries.jpg", 
             "/dbfs/mnt/training/dl/img/rose.jpg"]

img_paths = ["C:/Users/mikhail.galkin/Pictures/Saved Pictures/Images/Irish.jfif"]

from IPython.display import display
from PIL import Image

predict_images(img_paths, vgg16Model)

    
    















