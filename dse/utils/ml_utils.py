import os
import csv
import shutil
import hls4ml
import sys
import yaml
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from qkeras import *


## Function to create a simple MLP model
def create_model(args):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(args.inputs,), name='input1'))
    model.add(layers.Flatten(name='flatten1'))
    model.add(layers.Dense(args.outputs, name='dense1'))

    return model

## Function to save model architecture, weights, and configuration
def load_model_if_exists(args, name):
    model = None
    filename = f'{args.model_dir}/{name}-in{args.inputs:03d}-ou{args.outputs:03d}.h5'
    if os.path.exists(filename):
        model = load_model(filename)
    return model

def save_model(args, model, name):
    if name is None:
        name = model.name
    filename_prefix = f'{args.model_dir}/{name}-in{args.inputs:03d}-ou{args.outputs:03d}'
    model.save(f'{filename_prefix}.h5')
    model.save_weights(f'{filename_prefix}_weights.h5')
    with open(f'{filename_prefix}.json', 'w') as outfile:
        outfile.write(model.to_json())
    return


