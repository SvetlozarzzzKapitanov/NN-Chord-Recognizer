## Setup
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

## Train
.\.venv\Scripts\python.exe src\train.py

## Predict
.\.venv\Scripts\python.exe src\predict.py data\Em\Em-001.wav
