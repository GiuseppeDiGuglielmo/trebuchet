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
    pattern = r'ac_fixed<(\d+),(\d+),(true|false)>'
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
        if re.match(r'^  Design Total:', line, flags=re.IGNORECASE):
            values = line.split()[3:5]
            latency = values[0]
            ii = values[1]
        if re.match(r'^  Total Area Score:', line, flags=re.IGNORECASE):
            area = line.split()[-1]
            break
    return area, latency, ii


def get_rc_area(filename):
    with open(filename, 'r') as f:
        text = f.read()
        lines = text.splitlines()
        total_line_found = False
        for line in lines:
            if re.match(r'^--\s+END\s+Synthesis\s+area\s+report\s+for\s+design', line, flags=re.IGNORECASE):
                total_line_found = True
                break

        if total_line_found:
            data_line_index = lines.index(line) - 2
            words = lines[data_line_index].split()
            return str(words[2])
    return 0

def get_synthesis_time(filename, prefix='C/RTL'):
    # Regular expression to match the synthesis time line
    time_pattern = r'{} SYNTHESIS COMPLETED IN (\d+h)?(\d+m)?(\d+s)?'.format(prefix)

    # Initialize total synthesis time in seconds
    total_seconds = 0

    try:
        # Open the log file
        with open(filename, 'r') as file:
            # Read each line in the file
            for line in file:
                # Search for the time pattern in each line
                match = re.search(time_pattern, line)
                if match:
                    # Extract hours, minutes, and seconds from the match groups
                    hours = match.group(1)
                    minutes = match.group(2)
                    seconds = match.group(3)

                    print(f'{filename}/catapult.log {prefix} {hours} {minutes} {seconds}')

                    # Convert each time component to seconds and sum them up
                    if hours:
                        total_seconds += int(hours[:-1]) * 3600  # Remove 'h' and convert to seconds
                    if minutes:
                        total_seconds += int(minutes[:-1]) * 60  # Remove 'm' and convert to seconds
                    if seconds:
                        total_seconds += int(seconds[:-1])  # Remove 's'

    except FileNotFoundError:
        print(f'Error: The file {filename} does not exist.'.format(filename))
    except Exception as e:
        print(f'An error occurred: {e}')

    return total_seconds

