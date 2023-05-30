import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import tensorflow.keras as tk
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.simplefilter('ignore')

'''

define function to import images, 
convert them to an array, 
concatenate them

'''

def images_to_array(dir_path, 
                    file_type = 'png'):

    '''
    
    extract, convert and append to a list every image in dir_path.
    return it
    
    '''

    images_folder = []
    for file in os.listdir(dir_path):
        if file.endswith(file_type):
            img = Image.open(os.path.join(dir_path, file))
            img_array = np.array(img)
            images_folder.append(img_array)
    return images_folder

'''

call the function for both the images folders,
0 : 100 banana images,
101 : 200 glasses images.

concatenate those arrays to have an array with shape (200, 224, 224, 3)

'''

path_1 = '/Users/lucaberbenni/Desktop/repo/imageclassifier/output_folder/banana/'
path_2 = '/Users/lucaberbenni/Desktop/repo/imageclassifier/output_folder/glasses/'
images_array = np.concatenate((images_to_array(path_1), 
                               images_to_array(path_2)))

'''

define X, y to fit the model,
X is th entire images_array,
y a one dimensional array with 200 elements, 
1-100 = 0(banana), 
101-200 = 1(glasses)

define X_test importing images from test folder and converting them into array.
shape aspected (n-fotos(1), 224, 224, 3)

'''

X = images_array
y = np.array([0] * 100 + [1] * 100)

X_test = np.array(images_to_array('/Users/lucaberbenni/Desktop/repo/imageclassifier/output_folder/test/'))

'''

define the two categories in y

'''

y_cat = to_categorical(y)

'''

clear session and implement model architecture.

convolutional neural network
    -number of filters
    -size of filters
    -of how many pixels the filter moves in horizontal and vertical
    -no padding added
    -chose activation function (relu)
    -shape pictures
    -how to initialize weights of the filter

max pooling
    -downsampling feature map preserving most important function
    
    -size pooling window
    -movements pooling window in horizontal and vertical
    -no padded is added

flatten
    -reshape the array for hidden layer (1 dimensional array)

dense
    -layer

    -number of neurons
    -activation functions(relu, softmax at the end for classification problems(same number of neurons as the variables to classify))

'''

K.clear_session()
model = Sequential([
    
    Conv2D(filters = 64, 
           kernel_size = (3, 3), 
           strides = (3, 3), 
           padding = 'valid', 
           activation = tk.activations.relu, 
           input_shape = (224, 224, 3), 
           kernel_initializer = tk.initializers.GlorotNormal(seed = 34)), 

    MaxPooling2D(pool_size = (7, 7), 
                 strides = (3, 3), 
                 padding = 'valid'), 
                
    Conv2D(filters = 128, 
           kernel_size = (3, 3), 
           strides = (3, 3), 
           padding = 'valid', 
           activation = tk.activations.relu), 

    MaxPooling2D(pool_size = (7, 7), 
                 strides = (3, 3), 
                 padding = 'valid'), 

    Flatten(), 

    Dense(units = 80, 
          activation = tk.activations.relu), 

    Dense(units = 40, 
          activation = tk.activations.relu), 

    Dense(units = 2, 
          activation = tk.activations.softmax)
])

'''

show model summary

'''

summary = model.summary()

'''

compile()
    -configure the model for training

    -algorithm for optimization (adam)
    -loss function used to calculate errors
    -performance metric used to evaluate the model during training(accuracy)

fit()
    -model training

    -number of times to iterate over the dataset(backpropagation)
    -number of sample per gradient update
    -fraction of training data used for validation

'''

model.compile(optimizer = 'adam', 
              loss = tk.losses.categorical_crossentropy, 
              metrics = ['accuracy'])

history = model.fit(X, 
                    y_cat, 
                    epochs = 8, 
                    batch_size = 5, 
                    validation_split = 0.2)

'''

create a dataframe with history data and plot them

'''

df = pd.DataFrame(data = history.history)

plt.plot(df['loss'], label = 'Training Loss')
plt.plot(df['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(df['accuracy'], label = 'Training accuracy')
plt.plot(df['val_accuracy'], label = 'Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# prediction = model.predict(X_test)

# print(prediction.round(2))

# model.save('DeepLearning.h5')