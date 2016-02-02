#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST
This is a minimal example to write a feed-forward net.
"""
from __future__ import print_function
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import links as L
from chainer import optimizers
from chainer import serializers

import data
import net


#print('Load model from', "mlp.model")
model = L.Classifier(net.MnistMLP(784, 1000, 10))
serializers.load_hdf5("mlp.model", model)

serializers.save_npz("mlp.model.npz",model);

