This project is a deep learning image classifier implemented using TensorFlow and Keras. It includes two Python files that demonstrate different approaches to image classification using pre-trained models.

## Image Classification with Custom CNN

DeepLearning.py demonstrates image classification using a custom convolutional neural network (CNN) architecture. It follows the following steps:

1. Imports the necessary libraries, including Pandas, NumPy, Matplotlib, and PIL.

2. Defines a function to import images, convert them to arrays, and concatenate them.

3. Calls the function to import and preprocess images from two different folders, representing two categories: bananas and glasses.

4. Builds a CNN model architecture using TensorFlow and Keras. The architecture includes multiple convolutional layers, max pooling layers, a flattening layer, and fully connected dense layers.

5. Compiles the model by specifying the optimizer, loss function, and evaluation metric.

6. Trains the model using the imported and preprocessed image data.

7. Generates plots to visualize the training loss and accuracy over epochs.

8. Provides the option to make predictions on new, unseen images and save the trained model.

## Image Classification with Transfer Learning

TransferLearning.py demonstrates image classification using transfer learning with a pre-trained VGG16 model. It follows the following steps:

1. Imports the necessary libraries, including TensorFlow, Keras, Pandas, Matplotlib, and NumPy.

2. Loads the pre-trained VGG16 model and prints a summary of its architecture.

3. Modifies the pre-trained model by excluding the final fully connected layer.

4. Sets all layers in the modified model as non-trainable, except for the last five layers.

5. Loads and prepares images from the output folder for model training.

6. Builds a new Keras model by stacking the modified VGG16 base model with additional layers, including flatten and dense layers.

7. Compiles the new model by specifying the loss function, optimizer algorithm, and evaluation metric.

8. Trains the model using the prepared image data.

9. Generates plots to visualize the training loss and accuracy over epochs.

10. Provides the option to save the trained model for future use.

## Usage

1. Ensure that you have the required dependencies installed, including TensorFlow, Keras, pandas, matplotlib, and PIL.

2. Prepare your image data: Organize your images into separate directories based on their categories.

3. Choose the desired approach: Decide whether to use the custom CNN approach or the transfer learning approach.

4. Run the corresponding Python file: Execute the provided Python script (`DeepLearning.py` or `TransferLearning.py`) to import the images, preprocess them, build the model, train the model using the image data, and evaluate its performance.

5. Visualize the results: The script generates plots to visualize the training loss and accuracy. Use these plots to assess the model's performance and make any necessary adjustments.

6. (Optional) Make predictions or save the trained model: Depending on the chosen approach, you can enable the prediction functionality or save the trained model for future use.

## Future Enhancements

- Extend the prediction functionality: Implement the option to make predictions on new, unseen images using the trained models.

- Model persistence: Enhance the script to save and load the trained models for future use.

- Hyperparameter optimization: Incorporate techniques like grid search or random search to automatically find optimal hyperparameters for the models.
