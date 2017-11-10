# Neurons
A Python implementation of Neural Network without hidden layers. You can use it to train a neural network, and then use it to predict for new samples.

## Package structure
`neurons.py` provides the class to solve the problem. It follows the `scikit-learn` API. 

`func.py` is function library. Currently, it implements `sigmoid`, `tanh`, `relu` and their derivatives.

`gen.py` is used to generate the data. It takes in a configuration json file and several command line arguments. The directory used to store the data should already exist. Otherwise, it will reports error, saying can't find files or directories.

`demo.ipynb` shows how to use this package.

## Copyright
I, Wenjie ZHENG, am the only author of this package. All rights reserved.

## Contact
Please contact me if you need more information about this package.
