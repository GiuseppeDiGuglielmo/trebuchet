help:
	@ehco "INFO: make <tab> for targets"
.PHONY: help

jupyter-lab:
	jupyter-lab --no-browser --port=9998
.PHONY: jupyter-lab

ultraclean:
	rm -rf myproject_prj myproject_prj.tar.gz
.PHONY: ultraclean
