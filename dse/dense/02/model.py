# Disable some console warnings on the ASIC-group servers
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ml_utils import *
from hls_utils import *

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

def main(args):
    ## Define output files/locations:
    model_name = 'dense'
    proj_name = model_name

    W, I, S = parse_ac_fixed(args.precision)

    HLS4ML_PRJ_DIR = f'{args.project_dir}/hls4ml-{model_name}-in{args.inputs:03d}-ou{args.outputs:03d}-rf{args.reuse_factor:03d}-w{W:02d}i{I:02d}{S}-{args.iotype.replace("_", "")}-{args.strategy.lower()}_prj'
    TB_INPUT_FEATURES_DAT = f'{args.data_dir}/tb_input_features-in{args.inputs:03d}-ou{args.outputs:03d}.dat'
    TB_OUTPUT_PREDICTIONS_DAT = f'{args.data_dir}/tb_output_predictions-in{args.inputs:03d}-ou{args.outputs:03d}.dat'
    Y_TEST_LABELS_DAT = f'{args.data_dir}/y_test_labels-in{args.inputs:03d}-ou{args.outputs:03d}.dat'
    KERAS_JSON = f'{args.model_dir}/{model_name}-in{args.inputs:03d}-ou{args.outputs:03d}.json'
    KERAS_H5 = f'{args.model_dir}/{model_name}-in{args.inputs:03d}-ou{args.outputs:03d}_weights.h5'
    CONFIG_FILE_YML = f'{args.project_dir}/{proj_name}-in{args.inputs:03d}-ou{args.outputs:03d}-rf{args.reuse_factor:03d}-w{W:02d}i{I:02d}{S}-{args.iotype.replace("_", "")}-{args.strategy.lower()}_config.yml'
    CATAPULT_HLS4ML_TXT = f'{HLS4ML_PRJ_DIR}/{proj_name}_prj/{model_name}.v1/nnet_layer_results.txt'
    CATAPULT_RTL_RPT = f'{HLS4ML_PRJ_DIR}/{proj_name}_prj/{model_name}.v1/rtl.rpt'
    RTLCOMPILER_LOG = f'{HLS4ML_PRJ_DIR}/{proj_name}_prj/{model_name}.v1/rc.log'

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
    np.savetxt(TB_INPUT_FEATURES_DAT, x_test, fmt='%f')
    np.savetxt(TB_OUTPUT_PREDICTIONS_DAT, np.argmax(model.predict(x_test), axis=1), fmt='%d')
    np.savetxt(Y_TEST_LABELS_DAT, y_test, fmt='%d')

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
    config_ccs['OutputDir'] = HLS4ML_PRJ_DIR

    ## Model information files saved in the previous step
    config_ccs['KerasJson'] = KERAS_JSON
    config_ccs['KerasH5'] = KERAS_H5
    config_ccs['InputData'] = TB_INPUT_FEATURES_DAT
    config_ccs['OutputPredictions'] = TB_OUTPUT_PREDICTIONS_DAT

    ## General technology settings
    config_ccs['ClockPeriod'] = 10
    config_ccs['IOType'] = args.iotype
    config_ccs['ASICLibs'] = 'nangate-45nm_beh'
    config_ccs['FIFO'] = 'hls4ml_lib.mgc_pipe_mem'

    ## Post-training qunatization
    ## Global or per-layer configuration of precision, parallelism, etc
    config_ccs['HLSConfig'] = {
            'Model': {
                'Precision': args.precision,
                'ReuseFactor': args.reuse_factor,
                'Strategy': args.strategy
                },
            }

    ## Create .yml file
    print('============================================================================================')
    print(f'Writing YAML config file: {CONFIG_FILE_YML}')
    with open(CONFIG_FILE_YML, 'w') as yaml_file:
        yaml.dump(config_ccs, yaml_file, explicit_start=False, default_flow_style=False)

    print('\n============================================================================================')
    print('HLS4ML converting keras model/Catapult to HLS C++')
    hls_model_ccs = hls4ml.converters.keras_to_hls(config_ccs)
    hls_model_ccs.compile()

    if args.synth:
        print('============================================================================================')
        print('Synthesizing HLS C++ model using Catapult')
        hls_model_ccs.build(csim=True, synth=False, cosim=True, validation=False, vsynth=True, bup=False)
        # hls_model_ccs.build()

        ## Collect results from Catapult logs
        area_hls, latency_hls, ii_hls = get_hls_area_latency_ii_from_file(CATAPULT_RTL_RPT)

        ## TODO: Add here a function that collects Area, Latency, and II results from RTL compiler logs
        area_ls = get_rc_area(RTLCOMPILER_LOG)

        ## Print results on console
        print(model_name, args.inputs, args.outputs, args.reuse_factor, args.precision, area_hls, latency_hls, ii_hls, area_ls)

        ## Append results to CSV file
        file_exists = os.path.isfile('dse.csv')
        with open('dse.csv', 'a', newline='') as csvfile:
            fieldnames = ['Layer', 'IOType', 'Strategy', 'Inputs', 'Outputs', 'ReuseFactor', 'Precision', 'AreaHLS', 'LatencyHLS', 'IIHLS', 'AreaLS']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                    {
                        'Layer': model_name,
                        'IOType': args.iotype,
                        'Strategy': args.strategy,
                        'Inputs': args.inputs,
                        'Outputs': args.outputs,
                        'ReuseFactor': args.reuse_factor,
                        'Precision': args.precision,
                        'AreaHLS': area_hls,
                        'LatencyHLS': latency_hls,
                        'IIHLS': ii_hls,
                        'AreaLS': area_ls
                    })
    else:
        print('============================================================================================')
        print('Skipping HLS - To run Catapult directly:')
        print(f'cd {HLS4ML_PRJ_DIR}; catapult -file build_prj.tcl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure and convert the model for Catapult HLS')
    parser.add_argument('--inputs', type=int, default=8, help='Specify the Dense layer inputs')
    parser.add_argument('--outputs', type=int, default=2, help='Specify the Dense layer outputs')
    parser.add_argument('--reuse_factor', type=int, default=2, help='Specify the ReuseFactor value')
    parser.add_argument('--precision', type=str, default='ac_fixed<16,6,true>', help='Specify the Dense layer precision')
    parser.add_argument('--iotype', type=str, default='io_stream', help='Specify the layer interface (IOType)')
    parser.add_argument('--strategy', type=str, default='Resource', help='Specify the layer implementation (Strategy)')
    #
    parser.add_argument('--project_dir', type=str, default='hls4ml_prjs', help='Specify the base directory for the hls4ml project')
    parser.add_argument('--model_dir', type=str, default='models', help='Specify the base directory for the model files')
    parser.add_argument('--data_dir', type=str, default='data', help='Specify the base directory for the data files')
    #
    parser.add_argument('--synth', action='store_true', help='Specify whether to perform Catapult build and synthesis')
    args = parser.parse_args()

    main(args)
