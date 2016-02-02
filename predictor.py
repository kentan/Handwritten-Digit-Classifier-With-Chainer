#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST
This is a minimal example to write a feed-forward net.
"""

import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import links as L
from chainer import optimizers
from chainer import serializers

import net


#print('Load model from', "mlp.model")
model = L.Classifier(net.MnistMLP(784, 1000, 10))
serializers.load_npz("mlp.model.npz", model)

def prepare(value):
        value_len = len(value);
        for i in range(0,value_len):
           value[i] = np.float32(value[i]/255.0);
        rv = [[]];
        rv[0] = value;
        return rv;


def predict(input_data):
	prepared_data = prepare(input_data);
	x = chainer.Variable(np.asarray(prepared_data),volatile='on')
	y = model.predictor(x);
	return np.argmax(y.data);



