# Deep Neural Network Training and Testing for Vision-based Avoidance

The files in this folder are for training and testing a deep neural network for the vision-based avoidance methods defined in the high-level package.

## Overview

There are two key files that must be understood to produce a trained network.
* dnn_architecture
* train_dnn

### dnn_architecture

This file defines the deep neural network architecture.  It has 4 convolutional layers, then 3 fully connected layers.  Then the pose_inputs (currently just [x,y]) is attached to the top of the fully connected layer then there are 2 fully connected layers.

### train_dnn

In this file, you only need to worry about the ObstacleAvoidanceDataSet class, and then the functions:  main(), trainNet(), and testNet()

At main(), the model is initialized, and currently I send it to "cpu", even though I have it setup to send to "cuda" as well.

I define the three batch sizes for train, validation, and test.  Then I define the transforms.  Then the datasets are split and organized to the proper format.

Then the TrainNet function is called which starts to train the model using Adam Optimizer and Mean Squared Error Loss function.  The train loss should print after each batch and the validation loss prints after each Epoch.

The model is then saved.

## Requirements

The data must be in a folder under "/data" and it must contain the .csv file with all the image locations.
