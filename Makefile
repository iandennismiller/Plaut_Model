# PMSP

.PHONY: all
cd ..all:
	python3 code/simulator.py
	echo OK

.PHONY: requirements
requirements:
	pip install -r requirements.txt
