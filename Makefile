install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/preprocess.py
	python src/train.py

test:
	pytest -v