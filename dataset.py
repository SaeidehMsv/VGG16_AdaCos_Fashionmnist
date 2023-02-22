import pandas as pd
import tensorflow as tf
import numpy as np


def parse_input(x, y):
    return (x, y), y


def maketfdataset():
    train_data = pd.read_csv("fashion-mnist_train.csv")
    y_train = train_data['label']
    x_train = train_data.drop(columns='label')
    x_train = x_train / 255.0
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (-1, 28, 28))
    print(np.shape(x_train))
    test_data = pd.read_csv("fashion-mnist_test.csv")
    y_test = test_data['label']
    x_test = test_data.drop(columns='label')
    x_test = x_test / 255.0
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (-1, 28, 28))
    print(np.shape(x_test))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.map(parse_input)
    test_dataset = test_dataset.map(parse_input)
    return train_dataset.shuffle(100).batch(64), test_dataset.batch(64)
