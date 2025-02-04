@echo off
python -m pip install --upgrade pip

echo Creating fresh virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo Installing PyTorch...
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo Installing dependencies...
pip install sentencepiece protobuf==3.20.2 accelerate>=0.21.0
pip install transformers[torch]==4.41.0

echo Installing other requirements...
pip install pandas==2.1.4 openpyxl==3.1.2 scikit-learn==1.3.2 tqdm==4.66.1 numpy==1.26.2
pip install matplotlib==3.8.2 networkx==3.2.1 pillow==10.2.0

echo Verifying setup...
python verify_cuda.py

echo Starting GUI...
python section_classifier_gui.py

deactivate