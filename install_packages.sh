# 2) install PyTorch CUDA 12.1 wheels (self-contained, avoids ITT conflicts)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 3) install the rest
pip install transformers trl peft accelerate datasets wandb jupyterlab