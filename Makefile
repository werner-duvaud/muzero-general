#!/usr/bin/env make

.PHONY: train docker-build docker-console notebook install save-dependencies start console notebook-up attach-console tensorboard-train tensorboard-test

# ---------------------------------------------------------------------------------------------------------------------
# DEVELOPMENT
# ---------------------------------------------------------------------------------------------------------------------

MODIN_ENGINE=dask

# Will reproduce all stages to generate model based on changes
default:
	dvc repro && dvc push

# Will load data and models which specifaed by dvc files
checkout:
	dvc pull

train:
	echo "-m flag will run script in module mode, which accept relative imports"
	python -m src.train

test:
	echo "-m flag will run script in module mode, which accept relative imports"
	python -m src.test

# Will start notebook environment on http://0.0.0.0:8889
notebook: 
	jupyter notebook --ip=0.0.0.0 --allow-root --port=8889

install:
	pip install -r requirements.txt

# Will save all current dependencies to requirements.txt
save-dependencies:
	pip freeze > requirements.txt

# docs folder required for github pages
notebook-to-html:
	jupyter nbconvert ./*.ipynb --to html --output-dir="./docs"

notebook-to-python:
	jupyter nbconvert ./*.ipynb --to python --output-dir="./py_notebooks"

notebook-artifacts: notebook-to-html notebook-to-python chmod

metrics-diff:
	dvc metrics diff

tensorboard-train:
	tensorboard --logdir ./models/train/ --host 0.0.0.0

tensorboard-test:
	tensorboard --logdir ./models/eval/ --host 0.0.0.0

# ---------------------------------------------------------------------------------------------------------------------
# DOCKER
# ---------------------------------------------------------------------------------------------------------------------

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

start:
	docker-compose up --build

# Will start in docker develoment environment
docker-console:
	docker-compose run --service-ports muzero-notebook bash

console: docker-console

notebook-up:
	docker-compose up --build muzero-notebook

docker:
	docker-compose run -d --service-ports muzero-notebook

docker-it:
	docker-compose run --service-ports muzero-notebook 

attach-console:
	docker exec -it muzero-notebook bash

chmod:
	chmod -R 777 .

# ---------------------------------------------------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------------------------------------------------

gpu-monitor:
	watch -n 0.5 nvidia-smi

# ---------------------------------------------------------------------------------------------------------------------
# PYTHON PACKAGE
# ---------------------------------------------------------------------------------------------------------------------

package-dist:
	python setup.py sdist bdist_wheel

upload-to-test-repo:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload-package:
	twine upload dist/*

minor-bump:
	bump2version minor

patch-bump:
	bump2version patch

