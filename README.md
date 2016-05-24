# image_caption


This is an implementaton of an image caption retrieval system based 
on Keras deep learning library following my blog https://deeplearningmania.quora.com/
The motivation has been taken from the following paper http://arxiv.org/pdf/1511.06361v5.pdf 
Preprocessing is followed by https://github.com/ivendrov/order-embedding



Dependencies

This code is written in python. To use it you will need:

Python 2.7
Theano 0.7
Keras 1.0.3 
A recent version of NumPy and SciPy


Getting data

Download the dataset files (1 GB), including 10-crop VGG19 features, by running

wget http://www.cs.toronto.edu/~vendrov/order/coco.zip
