.PHONY: 
	install
	run

install:
	pip install -r requirements.txt

run:
	python3 Estimation.py