#用于将cifar10的数据可视化
# -*- coding: utf-8 -*-
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
import pickle

def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = p.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj['label_names']

def load_test_ten():
    label = load_CIFAR_Labels("./cifar10_data/cifar-10-batches-py/batches.meta")
    imgX, imgY = load_CIFAR_batch("./cifar10_data/cifar-10-batches-py/test_batch")
    imgX = imgX[0:10]
    imgY = imgY[0:10]
    return imgX, imgY, label

