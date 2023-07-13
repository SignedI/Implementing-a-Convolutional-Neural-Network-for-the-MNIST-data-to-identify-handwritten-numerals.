# Implementing-a-Convolutional-Neural-Network-for-the-MNIST-data-to-identify-handwritten-numerals.

In this project, I make a Convolutional Neural Network, and train it, so that it can identify different handwritten numerals.

The MNIST dataset, made initially by Stanford university in 2011, was used for this purpose. It consists of 60,000 training images and 10,000 testing images, each of which is a 28X28 pixel grayscale image with a label of its value added to it. Although all these images are single-digit numerals, the model can further be trained and optimised to detect all the numerals, as all of them are made from these single digits only.

In this model, we first reshape the training dataset according to the channel size, which we have kept 1. After that, we form the Convolutional Neural Network model using tensorflow and keras, and then train it using the MNIST dataset. The output that we get is the accuracy score with which it was able to identify the correct numerals.
