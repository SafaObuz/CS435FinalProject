#!/bin/bash
if [ -d "./venv" ]; then
    rm -rf venv
fi

python -3.12 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

# If needed, install CUDA-enabled PyTorch (adjust the command to match your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "Environment setup complete. To activate the environment in the future, run: source venv/bin/activate"
echo "Now checking if CUDA is enabled..."

python -c "import torch; print('CUDA device count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'No CUDA GPU detected')"