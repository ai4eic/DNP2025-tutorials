# Setup and Installation

## Overview

This chapter guides you through setting up your development environment for the DNP 2025 deep learning tutorials. We'll install Python, deep learning frameworks, and all necessary dependencies.

## System Requirements

### Hardware

**Minimum Requirements:**
- CPU: Modern multi-core processor (Intel Core i5 or equivalent)
- RAM: 8 GB
- Storage: 10 GB free space
- GPU: Optional (NVIDIA GPU with CUDA support recommended for faster training)

**Recommended:**
- CPU: Intel Core i7/i9 or AMD Ryzen 7/9
- RAM: 16 GB or more
- Storage: 20 GB+ SSD
- GPU: NVIDIA GPU with 6+ GB VRAM (RTX 2060 or better)

### Software

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.14+), or Windows 10/11 with WSL2
- **Python**: Version 3.8 or higher
- **Package Manager**: pip or conda

## Installation Options

We provide two installation methods:

1. **Conda Environment** (Recommended): Easiest and most reliable
2. **pip Virtual Environment**: Lightweight alternative

Choose one based on your preference.

## Option 1: Conda Installation

### Step 1: Install Anaconda or Miniconda

If you don't have conda installed:

**Linux/macOS:**
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts and restart terminal
```

**Windows:**
Download and run the installer from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### Step 2: Create Tutorial Environment

```bash
# Create new environment with Python 3.10
conda create -n dnp2025 python=3.10

# Activate environment
conda activate dnp2025
```

### Step 3: Install PyTorch (with GPU support)

For **NVIDIA GPU** with CUDA:
```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

For **CPU only**:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 4: Install Additional Packages

```bash
# Scientific computing and data analysis
conda install numpy scipy pandas matplotlib seaborn

# Machine learning utilities
conda install scikit-learn

# Jupyter notebooks
conda install jupyter jupyterlab ipywidgets

# Additional visualization tools
pip install plotly

# Physics and HEP tools (optional)
pip install uproot awkward particle
```

### Step 5: Install Tutorial Materials

```bash
# Clone the tutorial repository
git clone https://github.com/ai4eic/DNP2025-tutorials.git
cd DNP2025-tutorials

# Install any additional requirements
pip install -r requirements.txt
```

## Option 2: pip Virtual Environment

### Step 1: Install Python

Ensure Python 3.8+ is installed:
```bash
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv dnp2025-env

# Activate environment
# On Linux/macOS:
source dnp2025-env/bin/activate

# On Windows:
dnp2025-env\Scripts\activate
```

### Step 3: Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your configuration, then run the provided command. For example:

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Packages

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn jupyter jupyterlab ipywidgets plotly
```

### Step 5: Clone Repository

```bash
git clone https://github.com/ai4eic/DNP2025-tutorials.git
cd DNP2025-tutorials
pip install -r requirements.txt
```

## Verification

### Test Your Installation

Create a test script to verify everything is working:

```python
# test_installation.py
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
print("Matplotlib version:", plt.matplotlib.__version__)
print("scikit-learn version:", sklearn.__version__)

# Test simple computation
x = torch.rand(5, 3)
print("\nTest tensor:\n", x)

print("\n✓ All packages installed successfully!")
```

Run the test:
```bash
python test_installation.py
```

### Expected Output

You should see something like:
```
Python version: 3.10.x
NumPy version: 1.24.x
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3080
Matplotlib version: 3.x.x
scikit-learn version: 1.3.x

Test tensor:
 tensor([[0.xxxx, 0.xxxx, 0.xxxx],
        ...])

✓ All packages installed successfully!
```

## Download Tutorial Data

Tutorial datasets will be provided separately. To download:

```bash
# Navigate to tutorial directory
cd DNP2025-tutorials

# Download datasets (example - actual commands will be provided)
# Option 1: Direct download
wget https://example.com/fcal-tutorial-data.tar.gz
tar -xzf fcal-tutorial-data.tar.gz

# Option 2: From cloud storage
# Instructions will be provided during the tutorial
```

Expected data structure:
```
DNP2025-tutorials/
├── data/
│   ├── raw/               # Raw FCAL data
│   ├── processed/         # Preprocessed datasets
│   ├── train/            # Training sets
│   ├── val/              # Validation sets
│   └── test/             # Test sets
├── notebooks/            # Jupyter notebooks
├── scripts/              # Python scripts
└── models/               # Saved model checkpoints
```

## IDE Setup (Optional)

### Jupyter Lab

Launch Jupyter Lab for interactive development:
```bash
jupyter lab
```

### VS Code

For VS Code users:
1. Install Python extension
2. Install Jupyter extension
3. Select your conda/virtual environment as the Python interpreter
4. Open the tutorial folder

### PyCharm

For PyCharm users:
1. Open the tutorial project
2. Configure Python interpreter (Settings → Project → Python Interpreter)
3. Select your conda/virtual environment

## Troubleshooting

### CUDA Issues

**Problem**: CUDA not detected despite having NVIDIA GPU

**Solutions**:
- Update GPU drivers: [nvidia.com/drivers](https://www.nvidia.com/drivers)
- Verify CUDA installation: `nvcc --version`
- Reinstall PyTorch with correct CUDA version
- Check PyTorch-CUDA compatibility

### Package Conflicts

**Problem**: Dependency conflicts during installation

**Solutions**:
- Create fresh environment: `conda create -n dnp2025-clean python=3.10`
- Use specific package versions: `pip install package==version`
- Try conda-forge channel: `conda install -c conda-forge package`

### Memory Issues

**Problem**: Out of memory errors during training

**Solutions**:
- Reduce batch size in training scripts
- Use CPU if GPU memory is insufficient
- Enable gradient checkpointing for large models
- Close unnecessary applications

### Import Errors

**Problem**: `ModuleNotFoundError` when importing packages

**Solutions**:
- Ensure correct environment is activated
- Reinstall missing package: `pip install package_name`
- Check PYTHONPATH: `echo $PYTHONPATH`

## Getting Help

If you encounter issues:

1. Check the [FAQ](21-faq.md) section
2. Search [GitHub Issues](https://github.com/ai4eic/DNP2025-tutorials/issues)
3. Ask on the discussion forum
4. Contact tutorial organizers

## Next Steps

Once your environment is set up and verified:
- Continue to [FCAL Physics Background](03-fcal-physics.md) to learn about the detector
- Jump to [Understanding FCAL Data](04-data-understanding.md) to start working with data
- Explore example notebooks in the `notebooks/` directory

Your environment is ready for the tutorials! 🚀
