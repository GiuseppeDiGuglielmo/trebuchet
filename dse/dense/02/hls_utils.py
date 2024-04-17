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

## Function to parse Catapult results
def get_area_latency_from_file(filename, layer):
    with open(filename, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if layer in line:
            parts = line.split()
            area = parts[1]
            latency = parts[2]
            return area, latency
    return None, None


