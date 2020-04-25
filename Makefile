# PMSP

.PHONY: all
all:
	bin/pmsp.py run
	@echo OK

.PHONY: requirements
requirements:
	pip install -r requirements.txt
