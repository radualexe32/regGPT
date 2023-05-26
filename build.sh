#!/bin/sh
python3 -m venv env 
.\env\Scripts\activate
cd .\env
python3 -m pip install jupyter numpy pandas scikit-learn matplotlib openai torch torchvision torchaudio sklearn pytest-benchmark