import re
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

def parse_ac_fixed(input_string):
    pattern = r"ac_fixed<(\d+),(\d+),(true|false)>"
    match = re.search(pattern, input_string)

    if match:
        W = int(match.group(1))
        I = int(match.group(2))
        S = 's' if match.group(3) == 'true' else 'u'
        return W, I, S
    else:
        # Return None or raise an exception if no match is found
        return None

def get_hls_area_latency_ii_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        area, latency,ii = 0,0,0
    for line in lines:
        if re.match(r"^  Design Total:", line, flags=re.IGNORECASE):
            values = line.split()[3:5]
            latency = values[0]
            ii = values[1]
        if re.match(r"^  Total Area Score:", line, flags=re.IGNORECASE):
            area = line.split()[-1]
            break
    return area, latency, ii


def get_rc_area(filename):
    with open(filename, 'r') as file:
        text = file.read()
        lines = text.splitlines()
        total_line_found = False
        for line in lines:
            if re.match(r"^--\s+END\s+Synthesis\s+area\s+report\s+for\s+design", line, flags=re.IGNORECASE):
                total_line_found = True
                break

        if total_line_found:
            data_line_index = lines.index(line) - 2
            words = lines[data_line_index].split()
            return str(words[2])
    return 0


def  get_hls_ls_runtime(filename):
    
    hls_pattern = r"^.*\*\*\*\*\* C/RTL SYNTHESIS COMPLETED IN (\d+)h(\d+)m(\d+)s .*$"
    ls_pattern = r"^.*\*\*\*\*\* RC SYNTHESIS COMPLETED IN (\d+)h(\d+)m(\d+)s .*$"
     
    with open(log_file, 'r') as file:

        data_found = False
        values =[0,0]
        for line in file:

            hls_match = re.search(hls_pattern, line)
            ls_match = re.search(ls_pattern, line)

            if hls_match:
                hls_hours, hls_minutes, hls_seconds = map(int, hls_match.groups())
                hls_time= hls_hours * 3600 + hls_minutes * 60 + hls_seconds

            if ls_match :
                print(line)
                ls_hours, ls_minutes, ls_seconds = map(int, ls_match.groups())
                ls_time= ls_hours * 3600 + ls_minutes * 60 + ls_seconds
                return hls_time,ls_time
                
        return None, None
    
        


