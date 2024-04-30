MODEL = dense

help:
	@echo "INFO: make <TAB> to show targets"
.PHONY: help

#PYTHONPATH ?= ${MGC_HOME}/shared/pkgs/ccs_hls4ml/hls4ml
#PYTHONPATH ?= /asic/projects/C/CATAPULT/gdg/catapult_hls4ml_dev/2024.1_1/ccs_hls4ml/hls4ml
PYTHONPATH ?= /asic/projects/C/CATAPULT/gdg/catapult_hls4ml_dev/git/hls4ml

#
H ?= 8
W ?= 8
C ?= 4
O ?= 2
K := 3
P := "valid"
S := 1
#
IO ?= "io_parallel"
ST ?= "Latency"
RF ?= 1
PR ?= "ac_fixed<16,6,true>"

PROJECT_DIR ?= hls4ml_prjs
MODEL_DIR ?= models
DATA_DIR ?= data

run-ml:
	PYTHONPATH=$(PYTHONPATH) \
		python model.py \
			--height $(H) \
			--width $(W) \
			--channels $(C) \
			--outputs $(O) \
			--kernel $(K) \
			--padding $(P) \
			--stride $(S) \
			--iotype $(IO) \
			--strategy $(ST) \
			--reuse_factor $(RF) \
			--precision $(PR) \
			--project_dir $(PROJECT_DIR) \
			--model_dir $(MODEL_DIR) \
			--data_dir $(DATA_DIR)
.PHONY: run-ml

run-ml-hls:
	PYTHONPATH=$(PYTHONPATH) \
		python model.py \
			--height $(H) \
			--width $(W) \
			--channels $(C) \
			--outputs $(O) \
			--kernel $(K) \
			--padding $(P) \
			--stride $(S) \
			--iotype $(IO) \
			--strategy $(ST) \
			--reuse_factor $(RF) \
			--precision $(PR) \
			--project_dir $(PROJECT_DIR) \
			--model_dir $(MODEL_DIR) \
			--data_dir $(DATA_DIR) \
			--synth
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

