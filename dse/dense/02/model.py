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

def main(args):
    ## TODO: move this hardcoded knobs to the interface so we can use the flags
    ## --iotype "io_stream"
    ## --strategy "Resource"
    ## Define some knobs
    io_type = 'io_stream'
    strategy = 'Resource'

    ## Define output files/location:
    model_name = 'dense'
    proj_name = model_name
    out_dir = f'{args.project_dir}/my-Catapult-{model_name}-in{args.inputs:03d}-ou{args.outputs:03d}-rf{args.reuse_factor:03d}-{io_type}-{strategy}'

    ## Determine the directory containing this model.py script in order to locate the associated .dat file
    sfd = os.path.dirname(__file__)

    ## Tiny MLP (it does really nothing)
    num_samples = 1000

    ## Generate random feature data
    x_train = np.random.rand(num_samples, args.inputs)
    x_test = np.random.rand(num_samples, args.inputs)

    ## Generate random labels (0 or 1) and convert to one-hot encoded format
    y_train = np.random.randint(args.outputs, size=(num_samples, 1))
    y_train = keras.utils.to_categorical(y_train, args.outputs)
    y_test = np.random.randint(args.outputs, size=(num_samples, 1))
    y_test = keras.utils.to_categorical(y_test, args.outputs)

    ## Load previously trained model
    model = load_model_if_exists(args, model_name)

    ## If it does not exist, then train
    if (model == None):
        ## Create and compile the model
        model = create_model(args)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        ## Train the model
        model.fit(x_train, y_train, epochs=10)

    ## Save input features and model predictions
    np.savetxt(f'{args.data_dir}/tb_input_features-in{args.inputs:03d}-ou{args.outputs:03d}.dat', x_test, fmt='%f')
    np.savetxt(f'{args.data_dir}/tb_output_predictions-in{args.inputs:03d}-ou{args.outputs:03d}.dat', np.argmax(model.predict(x_test), axis=1), fmt='%d')
    np.savetxt(f'{args.data_dir}/y_test_labels-in{args.inputs:03d}-ou{args.outputs:03d}.dat', y_test, fmt='%d')

    save_model(args, model, name=model_name)
    print(hls4ml.__version__)

    ## Uncomment the following lines to list the default configuration
    #config_ccs = hls4ml.utils.config_from_keras_model(model, granularity='name')
    #print(json.dumps(config_ccs, indent=4))

    ## Configure and convert the model for Catapult HLS

    ## Start with an empty configuration. See above for default.
    config_ccs = {}

    ## General project settings
    config_ccs['Backend'] = 'Catapult'
    config_ccs['ProjectName'] = proj_name
    config_ccs['OutputDir'] = out_dir

    ## Model information files saved in the previous step
    config_ccs['KerasJson'] = f'{args.model_dir}/{model_name}-in{args.inputs:03d}-ou{args.outputs:03d}.json'
    config_ccs['KerasH5'] = f'{args.model_dir}/{model_name}-in{args.inputs:03d}-ou{args.outputs:03d}_weights.h5'
    config_ccs['InputData'] = f'{args.data_dir}/tb_input_features-in{args.inputs:03d}-ou{args.outputs:03d}.dat'
    config_ccs['OutputPredictions'] = f'{args.data_dir}/tb_output_predictions-in{args.inputs:03d}-ou{args.outputs:03d}.dat'

    ## General technology settings
    config_ccs['ClockPeriod'] = 10
    config_ccs['IOType'] = io_type
    config_ccs['ASICLibs'] = 'saed32rvt_tt0p78v125c_beh'
    config_ccs['FIFO'] = 'hls4ml_lib.mgc_pipe_mem'

    ## Post-training qunatization
    ## Global or per-layer configuration of precision, parallelism, etc
    config_ccs['HLSConfig'] = {
            'Model': {
                'Precision': args.precision,
                'ReuseFactor': args.reuse_factor,
                'Strategy': strategy
                },
            }

    ## Create .yml file
    print("============================================================================================")
    print('Writing YAML config file: '+proj_name+'_config.yml')
    with open(f'{args.project_dir}/{proj_name}-in{args.inputs:03d}-ou{args.outputs:03d}-rf{args.reuse_factor:03d}-{io_type}-{strategy}_config.yml', 'w') as yaml_file:
        yaml.dump(config_ccs, yaml_file, explicit_start=False, default_flow_style=False)

    print("\n============================================================================================")
    print("HLS4ML converting keras model/Catapult to HLS C++")
    hls_model_ccs = hls4ml.converters.keras_to_hls(config_ccs)
    hls_model_ccs.compile()

    if args.synth:
        print("============================================================================================")
        print("Synthesizing HLS C++ model using Catapult")
        hls_model_ccs.build(csim=True, synth=True, cosim=True, validation=False, vsynth=False, bup=False)
        # hls_model_ccs.build()

        ## Collect results from Catapult logs
        result_file = f'{out_dir}/{proj_name}_prj/{model_name}.v1/nnet_layer_results.txt'
        area, latency = get_area_latency_from_file(result_file, model_name)

        ## Print results on console
        print(model_name, args.inputs, args.outputs, args.reuse_factor, args.precision, area, latency)

        ## Append results to CSV file
        file_exists = os.path.isfile('dse.csv')
        with open('dse.csv', 'a', newline='') as csvfile:
            fieldnames = ['Layer', 'IOType', 'Strategy', 'Inputs', 'Outputs', 'ReuseFactor', 'Precision', 'Area', 'Latency']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                    {
                        'Layer': model_name,
                        'IOType': io_type,
                        'Strategy': strategy,
                        'Inputs': args.inputs,
                        'Outputs': args.outputs,
                        'ReuseFactor': args.reuse_factor,
                        'Precision': args.precision,
                        'Area': area,
                        'Latency': latency
                    })
    else:
        print("============================================================================================")
        print("Skipping HLS - To run Catapult directly:")
        print('cd ' + out_dir + '; catapult -file build_prj.tcl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure and convert the model for Catapult HLS')
    parser.add_argument('--reuse_factor', type=int, default=2, help='Specify the ReuseFactor value')
    parser.add_argument('--precision', type=str, default='ac_fixed<16,6,true>', help='Specify the Dense layer precision')
    parser.add_argument('--inputs', type=int, default=8, help='Specify the Dense layer inputs')
    parser.add_argument('--outputs', type=int, default=2, help='Specify the Dense layer outputs')
    parser.add_argument('--synth', action='store_true', help='Specify whether to perform Catapult build and synthesis')
    parser.add_argument('--project_dir', type=str, default='hls4ml_prjs', help='Specify the base directory for the hls4ml project')
    parser.add_argument('--model_dir', type=str, default='models', help='Specify the base directory for the model files')
    parser.add_argument('--data_dir', type=str, default='data', help='Specify the base directory for the data files')
    args = parser.parse_args()

    main(args)
