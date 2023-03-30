import os
import math
import time
import random
import inspect
import hashlib
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *

class LinearRegressionKeras(Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        # only generate a single scalar output, so set the parameter to 1
        self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        
    def forward(self, X):
        # invoke the built-in __call__ method of the predefined layers to compute the outputs.
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)
    
    def get_w_b(self):
        return self.get_weights()[0], self.get_weights()[1]