# Disable some console warnings on the ASIC-group servers
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

#from utils.ml_utils import *
from utils.hls_utils import *

import csv
import math
import shutil
import hls4ml
import sys
import yaml
import json
import pickle
import argparse
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Flatten, SeparableConv2D

from sklearn.metrics import accuracy_score
from qkeras import *

def create_model(args):
    model = Sequential()
    model.add(Input(shape=(args.height, args.width, args.channels), name='input1'))
    model.add(SeparableConv2D(filters=args.outputs, kernel_size=(args.kernel, args.kernel), padding=args.padding, name='sepconv1'))
    return model

def load_model_if_exists(args, filename):
    model = None
    if os.path.exists(filename):
        model = load_model(filename)
    return model

def save_model(model, model_json, model_weights_h5, model_h5):
    model.save(model_h5)
    model.save_weights(model_weights_h5)
    with open(model_json, 'w') as outfile:
        outfile.write(model.to_json())

def main(args):
    ## Define output files/locations:
    model_name = 'sepconv2d'
    proj_name = model_name

    W, I, S = parse_ac_fixed(args.precision)

    DESIGN_POINT = f'-h{args.height:03d}'
    DESIGN_POINT += f'-w{args.width:03d}'
    DESIGN_POINT += f'-c{args.channels:03d}'
    DESIGN_POINT += f'-o{args.outputs:03d}'
    DESIGN_POINT += f'-k{args.kernel:03d}'
    DESIGN_POINT += f'-s{args.stride:03d}'
    DESIGN_POINT += f'-{args.padding}'

    HLS_DESIGN_POINT = DESIGN_POINT
    HLS_DESIGN_POINT += f'-rf{args.reuse_factor:03d}'
    HLS_DESIGN_POINT += f'-w{W:02d}i{I:02d}{S}'
    HLS_DESIGN_POINT += f"-{args.iotype.replace('_', '')}"
    HLS_DESIGN_POINT += f'-{args.strategy.lower()}'

    TB_INPUT_FEATURES_DAT = f'{args.data_dir}/tb_input_features{DESIGN_POINT}.dat'
    TB_OUTPUT_PREDICTIONS_DAT = f'{args.data_dir}/tb_output_predictions{DESIGN_POINT}.dat'
    Y_TEST_LABELS_DAT = f'{args.data_dir}/y_test_labels{DESIGN_POINT}.dat'
    KERAS_JSON = f'{args.model_dir}/{model_name}{DESIGN_POINT}.json'
    KERAS_WEIGHTS_H5 = f'{args.model_dir}/{model_name}{DESIGN_POINT}_weights.h5'
    KERAS_H5 = f'{args.model_dir}/{model_name}{DESIGN_POINT}.h5'

    HLS4ML_PRJ_DIR = f'{args.project_dir}/hls4ml-{model_name}{HLS_DESIGN_POINT}_prj'
    CONFIG_FILE_YML = f'{args.project_dir}/{proj_name}{HLS_DESIGN_POINT}_config.yml'

    CATAPULT_HLS4ML_TXT = f'{HLS4ML_PRJ_DIR}/{proj_name}_prj/{model_name}.v1/nnet_layer_results.txt'
    CATAPULT_RTL_RPT = f'{HLS4ML_PRJ_DIR}/{proj_name}_prj/{model_name}.v1/rtl.rpt'
    RTLCOMPILER_LOG = f'{HLS4ML_PRJ_DIR}/{proj_name}_prj/{model_name}.v1/rc.log'
    CATAPULT_LOG = f'{HLS4ML_PRJ_DIR}/catapult.log'

    ## Determine the directory containing this model.py script in order to locate the associated .dat file
    sfd = os.path.dirname(__file__)

    ## The model does really nothing, random data...
    num_samples = 1000

    ## Output sizes
    if args.padding == 'same':
        padding = max(0, math.floor((args.kernel-args.stride)/2))
    else:
        padding = 0
    out_height = math.floor((args.height-args.kernel+2*padding)/args.stride)+1
    out_width = math.floor((args.width-args.kernel+2*padding)/args.stride)+1

    ## Generate random input data
    x_train = np.random.rand(num_samples, args.height, args.width, args.channels)
    x_test = np.random.rand(num_samples, args.height, args.width, args.channels)

    ## Generate random output data
    y_train = np.random.randint(args.outputs, size=(num_samples, out_height, out_width, args.outputs))
    y_test = np.random.randint(args.outputs, size=(num_samples, out_height, out_width, args.outputs))

    ## Load previously trained model
    model = load_model_if_exists(args, KERAS_H5)

    ## If it does not exist, then train
    if (model == None):
        ## Create and compile the model
        model = create_model(args)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        ## Train the model
        model.fit(x_train, y_train, epochs=10)

    ## Save input features and model predictions
    np.savetxt(TB_INPUT_FEATURES_DAT, x_test.reshape(num_samples, -1), fmt='%f')
    np.savetxt(TB_OUTPUT_PREDICTIONS_DAT, model.predict(x_test).reshape(num_samples, -1), fmt='%f')
    np.savetxt(Y_TEST_LABELS_DAT, y_test.reshape(num_samples, -1), fmt='%f')

    save_model(model, KERAS_JSON, KERAS_WEIGHTS_H5, KERAS_H5)

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
    config_ccs['ProjectDir'] = f'{proj_name}_prj' #TODO This seems to be part of the Catapult backend, but not in Vivado?
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

    ## Post-training quantization
    ## Global or per-layer configuration of precision, parallelism, etc
    config_ccs['HLSConfig'] = {
            'Model': {
                'Precision': args.precision,
                'ReuseFactor': args.reuse_factor,
                'Strategy': args.strategy
                },
            }

    ### Create .yml file
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
        hls_model_ccs.build(csim=True, synth=True, cosim=True, validation=False, vsynth=True, bup=False)
        # hls_model_ccs.build()

        ### Collect results from Catapult logs
        #area_hls, latency_hls, ii_hls = get_hls_area_latency_ii_from_file(CATAPULT_RTL_RPT)

        ### Collect results from RTL compiler logs
        #area_syn = get_rc_area(RTLCOMPILER_LOG)

        ### Collect Catapult runtime
        #runtime_hls = get_synthesis_time(CATAPULT_LOG, 'C/RTL')

        ### Collect RC runtime
        #runtime_syn = get_synthesis_time(CATAPULT_LOG, 'RC')

        ### Print results on console
        #print(model_name, args.height, args.width, args.channels, args.outputs, args.kernel, args.padding, args.stride, args.reuse_factor, args.precision, area_hls, latency_hls, ii_hls, area_syn, runtime_hls, runtime_syn)

        ### Append results to CSV file
        #file_exists = os.path.isfile('dse.csv')
        #with open('dse.csv', 'a', newline='') as csvfile:
        #    fieldnames = ['Layer', 'IOType', 'Strategy', 'Height', 'Width', 'Channels', 'Outputs', 'Kernel', 'Padding', 'Stride', 'Outputs', 'ReuseFactor', 'Precision', 'AreaHLS', 'LatencyHLS', 'IIHLS', 'AreaSYN', 'RuntimeHLS', 'RuntimeSYN']
        #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #    if not file_exists:
        #        writer.writeheader()
        #    writer.writerow(
        #            {
        #                'Layer': model_name,
        #                'IOType': args.iotype,
        #                'Strategy': args.strategy,
        #                'Height': args.height,
        #                'Width': args.width,
        #                'Channels': args.channels,
        #                'Outputs': args.outputs,
        #                'Kernel': args.kernel,
        #                'Padding': args.padding,
        #                'Stride': args.stride,
        #                'ReuseFactor': args.reuse_factor,
        #                'Precision': args.precision,
        #                'AreaHLS': area_hls,
        #                'LatencyHLS': latency_hls,
        #                'IIHLS': ii_hls,
        #                'AreaSYN': area_syn,
        #                'RuntimeHLS': runtime_hls,
        #                'RuntimeSYN': runtime_syn
        #            })
    else:
        print('============================================================================================')
        print('Skipping HLS - To run Catapult directly:')
        print(f'cd {HLS4ML_PRJ_DIR}; catapult -file build_prj.tcl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure and convert the model for Catapult HLS')
    #
    parser.add_argument('--height', type=int, default=8, help='Specify the SeparableConv2D layer input height')
    parser.add_argument('--width', type=int, default=8, help='Specify the SeparableConv2D layer input width')
    parser.add_argument('--channels', type=int, default=8, help='Specify the SeparableConv2D layer input channels')
    parser.add_argument('--outputs', type=int, default=2, help='Specify the Dense layer outputs')
    parser.add_argument('--kernel', type=int, default=3, help='Specify the SeparableConv2D layer kernel size')
    parser.add_argument('--padding', type=str, default='valid', help='Specify the SeparableConv2D layer padding (same, valid)')
    parser.add_argument('--stride', type=int, default=1, help='Specify the SeparableConv2D layer kernel stride')
    #
    parser.add_argument('--iotype', type=str, default='io_stream', help='Specify the layer interface (IOType)')
    parser.add_argument('--strategy', type=str, default='Resource', help='Specify the layer implementation (Strategy)')
    parser.add_argument('--reuse_factor', type=int, default=2, help='Specify the ReuseFactor value')
    parser.add_argument('--precision', type=str, default='ac_fixed<16,6,true>', help='Specify the Dense layer precision')
    #
    parser.add_argument('--project_dir', type=str, default='hls4ml_prjs', help='Specify the base directory for the hls4ml project')
    parser.add_argument('--model_dir', type=str, default='models', help='Specify the base directory for the model files')
    parser.add_argument('--data_dir', type=str, default='data', help='Specify the base directory for the data files')
    #
    parser.add_argument('--synth', action='store_true', help='Specify whether to perform Catapult build and synthesis')
    args = parser.parse_args()

    main(args)
