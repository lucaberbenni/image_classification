import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow import keras
import numpy as np
import pandas as pd
import os

'''

load the pre-trained VGG16 model using keras, 
print a summary of the model architecture.

weights:
    -load pretrained weights from imagenet dataset
    -the model will be initialized with pre-trained weights from imagenet dataset

'''

vgg_model = keras.applications.vgg16.VGG16(weights = 'imagenet')
vgg_model.summary()

'''

modify the pre-trained model

include_top:
    -the final fully connected layer of the network won't be included in the model

'''
base_model = keras.applications.vgg16.VGG16(weights = 'imagenet', 
                                            input_shape = (224, 224, 3), 
                                            include_top = False)
base_model.summary()

'''

set all the layer on the pre-trained model as non-trainable.
the weights of the pre-trained VGG16 model will be fixed and won't be updated

print a summary of the based model architecture

'''
for layers in base_model.layers[:]:
    layers.trainable = False
base_model.summary()

'''

unfreeze the last 5 layers, 
allowing them to be trained during fine-tuning

'''

unfreeze_layers = 5
for layers in base_model.layers[-unfreeze_layers:]:
    layers.trainable = True

'''
set the base path where the output folder is located, 
get the list of subfolders in the output folder, except the first three items

'''

base_path = '/Users/lucaberbenni/Desktop/repo/imageclassifier/output_folder/'
classes = os.listdir(base_path)[3 : ]

'''

load images from output folder and prepares them for model training

'''
def load_image(base_path):

    '''
    
    create empty lists for the input images and their corresponding labels
    
    '''
    X_list = []
    y_list = []

    '''
    
    get the list of subfolders in the output folder
    
    '''

    classes = os.listdir(base_path)[3 : ]

    '''
    
    loop through each class (subfolder) and each image file whitin that class
    
    '''

    for class_ in classes:
        files = os.listdir(base_path + class_)
        for file in files:

            '''
            
            load the image file and resize it to 224x224 pixels
            
            '''

            pic = keras.preprocessing.image.load_img(path = base_path + class_ + '/' + f'{file}', 
                                                     target_size = (224, 224))
            
            '''
            
            convert the image to an array and apply preprocess
            
            '''

            numpy_image = np.array(pic)
            processed_image = preprocess_input(numpy_image)

            '''
            
            add the preprocessed image and its corresponding label to the input and label list
            
            '''

            X_list.append(processed_image)
            y_list.append(class_)

    '''
    
    convert input and label list to np.array
    
    '''

    X = np.array(X_list)
    y = np.array(y_list)
    
    '''
    
    shuffle the input and label arrays in the same way
    
    '''

    shuffler =np.random.permutation(len(X))
    X = X[shuffler]
    y = y[shuffler]

    '''
    
    return input, label and classes arrays
    
    '''

    return X, y, classes

'''

load the images and their corresponding label through load_image() function

create a pandas series object from y using pd.Series()
map class names to integer labels using a dictionary comprehension

prepare the labels for use in training a model using to categorical()
one-hot-encodes the integer labels
convert every label in to a binary vector, where the index corresponding to label value is set to 1 and all other values are set to 0

'''

X, y, classes = load_image(base_path)
y_series = pd.Series(y).map({classes[0] : 0, classes[1] : 1})
y = to_categorical(y_series)

'''

create a new keras model by stacking the pre-trained VGG16 base model with a new model consisting in:
    -flatten layer
    -two dense layers with 100 units each and relu activation function
    -softmax activation
    
clear_session() to remove any previous defined models or layers

the show new_model summary

flatten() to flatten the input into 2D
dense() fully connected layer with 100 unit and relu as activation function

'''

K.clear_session()
new_model = keras.models.Sequential([
    base_model, 
    keras.layers.Flatten(), 
    keras.layers.Dense(units=100, 
                       activation=keras.activations.relu, 
                       name='fc1'), 
    keras.layers.Dense(units=100, 
                       activation=keras.activations.relu, 
                       name='fc2'),
    keras.layers.Dense(units = 2, 
                       activation = keras.activations.softmax, 
                       name = 'output')
])
new_model.summary()

'''

compile a keras module

loss to choose the loss function, 
categorical_crossenentropy good for multi-class classification problems

adam as optimizer algorithm, with a learning rate of 0,008

metrics for the evaluation metric used by the test
CategoricalAccuracy used in this  case(good for classification problems)

'''

new_model.compile(loss = keras.losses.categorical_crossentropy, 
                  optimizer = keras.optimizers.Adam(learning_rate = 0.008), 
                  metrics = keras.metrics.CategoricalAccuracy())

'''

train the keras model

X as input datas
y as target data

batch size as number of training examples the model should see at once before updating the weights
bigger batch size model faster but less precise

epochs number of iteration over the entire dataset used for training

validation_split specifies the fraction of training data used for validation

'''

history = new_model.fit(
    X, 
    y, 
    batch_size = 21, 
    epochs = 15, 
    validation_split = 0.2
)

# new_model.save('TransferLearning.h5')

df = pd.DataFrame(data = history.history)

plt.plot(df['loss'], label = 'Training Loss')
plt.plot(df['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(df['categorical_accuracy'], label = 'Training accuracy')
plt.plot(df['val_categorical_accuracy'], label = 'Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()