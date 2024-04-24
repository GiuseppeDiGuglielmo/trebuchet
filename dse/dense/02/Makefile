MODEL = dense

help:
	@echo "INFO: make <TAB> to show targets"
.PHONY: help

#PYTHONPATH ?= ${MGC_HOME}/shared/pkgs/ccs_hls4ml/hls4ml
#PYTHONPATH ?= /asic/projects/C/CATAPULT/gdg/catapult_hls4ml_dev/2024.1/ccs_hls4ml/hls4ml
PYTHONPATH = /asic/projects/C/CATAPULT/gdg/catapult_hls4ml_dev/git/hls4ml

IO ?= "io_stream"
ST ?= "Resource"
RF ?= 1
IN ?= 8
OU ?= 2

PADDED_RF = $(shell printf '%03d' $(RF))
PADDED_IN = $(shell printf '%03d' $(IN))
PADDED_OU = $(shell printf '%03d' $(OU))

PROJECT_DIR ?= hls4ml_prjs
MODEL_DIR ?= models
DATA_DIR ?= data

run-ml:
	PYTHONPATH=$(PYTHONPATH) \
		python model.py \
			--iotype $(IO) \
			--strategy $(ST) \
			--reuse_factor $(RF) \
			--inputs $(IN) \
			--outputs $(OU) \
			--project_dir $(PROJECT_DIR) \
			--model_dir $(MODEL_DIR) \
			--data_dir $(DATA_DIR)
.PHONY: run-ml

run-ml-hls:
	PYTHONPATH=$(PYTHONPATH) \
		python model.py \
			--synth \
			--iotype $(IO) \
			--strategy $(ST) \
			--reuse_factor $(RF) \
			--inputs $(IN) \
			--outputs $(OU) \
			--project_dir $(PROJECT_DIR) \
			--model_dir $(MODEL_DIR) \
			--data_dir $(DATA_DIR)
.PHONY: run-ml-hls

run-dse:
	./run_dse.sh
.PHONY: run-dse

jupyter:
	PYTHONPATH=$(PYTHONPATH) \
		jupyter notebook --no-browser --port=8887
.PHONY: jupyter

clean:
	rm -rf hls4ml_prjs/*_prj*
	rm -f hls4ml_prjs/*.yml
.PHONY: clean

ultraclean: clean
	rm -rf .ipynb_checkpoints
	rm -f ccs_hls_model_predictions.dat
	rm -f $(MODEL).h5
	rm -f $(MODEL).json
	rm -f $(MODEL)_weights.h5
	rm -f $(MODEL)_config.yml
	rm -f tb_input_features.dat
	rm -f tb_output_predictions.dat
	rm -f y_test_labels.dat
	rm -rf __pycache__
.PHONY: clean

