#High-level Project Overview
This is Udacity Data Scientist Nano Degree, Capstone dog breed classifier project. This project uses Convolutional Neural Networks (CNNs). The aim of this project is to build an algorithm to process real-world, user-supplied images. When a user provides an image of a dog, the algorithm will identify the dog’s breed. If image of a human is provided, the it will identify the dog breed it resembles.

In order to provide a solution for this objective following approach is taken:

-Step 0: Import Packages and Datasets
-Step 1: Detect Human Face
-Step 2: Detect Dogs
-Step 3: Create a CNN to classify Dog Breeds (from scratch)
-Step 4: Use a CNN to classify Dog Breeds (using Transfer Learning)
-Step 5: Create a CNN to classify Dog Breeds (using Transfer Learning)
-Step 6: Write an algorithm
-Step 7: Test algorithm

##— — — — — — — Description of Input Data — — — — — — — — — — -
Step 0 :
Import Datasets- Data sets are provided by Udacity for both human and Dogs. To start datasets to train, test, and validation is loaded.

a) train_files, valid_files, test_files - numpy arrays containing file paths to images.

b)train_targets, valid_targets, test_targets - numpy arrays containing onehot-encoded classification labels.

c)dog_names - list of string-valued dog breed names for translating labels.
##— — — — — — — — — —Approach/Strategy — — — — — — —
Step 1: Detect Human Face-
Load Human data set, there are 13233 total human images.
