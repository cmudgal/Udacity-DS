# High-level Project Overview
This is Udacity Data Scientist Nano Degree, Capstone dog breed classifier project. This project uses Convolutional Neural Networks (CNNs). The aim of this project is to build an algorithm to process real-world, user-supplied images. When a user provides an image of a dog, the algorithm will identify the dog’s breed. If image of a human is provided, the it will identify the dog breed it resembles.

In order to provide a solution for this objective following approach is taken:

- Step 0: Import Packages and Datasets
- Step 1: Detect Human Face
- Step 2: Detect Dogs
- Step 3: Create a CNN to classify Dog Breeds (from scratch)
- Step 4: Use a CNN to classify Dog Breeds (using Transfer Learning)
- Step 5: Create a CNN to classify Dog Breeds (using Transfer Learning)
- Step 6: Write an algorithm
- Step 7: Test algorithm

## — — — — — — — Description of Input Data — — — — — — — — — — -
Step 0 :
Import Datasets- Data sets are provided by Udacity for both human and Dogs. To start datasets to train, test, and validation is loaded.

a) train_files, valid_files, test_files - numpy arrays containing file paths to images.

b)train_targets, valid_targets, test_targets - numpy arrays containing onehot-encoded classification labels.

c)dog_names - list of string-valued dog breed names for translating labels.
## — — — — — — — — — —Approach/Strategy — — — — — — —
## Step 1: Detect Human Face-
Load Human data set, there are 13233 total human images.
The face detector algorithm assessment shows that it is able to detect human faces with 100% accuracy and failed 11% when encountered with dog images.

## Step 2 Detect Dogs:
This step uses pre-trained ResNet-50 model to detect dogs in images. ResNet-50 model is downloaded, along with weights that have been trained on ImageNet. ImageNet is a very large, dataset used for image classification and other vision tasks. Given an image, this pre-trained ResNet-50 model returns a prediction for the object that is contained in the image.
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
CNN Convolutional Neural Network is a class of deep neural network. It is made of neurons with learnable weights and biases.
CNNs have wide variety of applications in image and video recognition, recommender systems and natural language processing.
A CNN consists of an input and an output layer, along with multiple hidden layers.The hidden layers consist of a series of convolutional layers that convolve with a multiplication or other dot product.

## Step 4: Use a CNN to Classify Dog Breeds
Train a CNN using transfer learning. Obtain Bottleneck Features
The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.

## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Use bottleneck features from pretrained Resnet50 model(provided by Udacity) to build our CNN to identify dog breed from images.
Above trained model has test accuracy of 81.69% which is better than previous model which are trained using VGG16 or CNN(without tranfer learning).
- Model
- Compile
- Train
- Load
- Test

Above trained model has test accuracy of 81.69% which is better than previous model which are trained using VGG16 or CNN(without tranfer learning).
## Step 6: Write your Algorithm
Determine whether the image contains a human, dog, or neither.

if a dog is detected in the image, return the predicted breed.
if a human is detected in the image, return the resembling dog breed.
if neither is detected in the image, provide output that indicates an error.
## Step 7: Test Your Algorithm
Plotting the data through this algorithm predicted human images with the dog breed with their resemblance and returned error if the image was not dog or human.

## — — — — — — — — — — —Reflection — — — — — — — — — — — — —
Convolutional layers are memory intensive and, therefore need more computation to store the data.
In contrast, in the densely connected layer of softmax, little memory space is needed. But, requires numerous parameters to achieve learning.
It is important to know the sizes of the data and the parameters because, there are many layer in convolutional neural networks, and these values can shoot exponentially.
## — — — — — — — — — — — -Improvement — — — — — — — — — — — —
If below techniques can be used to further improve the model results:

- Increasing the training time/epochs
- Data size
- Hyperparameters Tuning
## — — — — — — — — — — — — — -Acnowledgement — — — — — — — — —
- Udacity — https://udacity.com
- Keras — https://keras.io
- Kaggle — https://kaggle.com
