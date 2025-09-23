# Installation Guide

This page provides detailed instructions for installing and setting up HIDISC on your system.

## Prerequisites

Before installing HIDISC, ensure you have the following prerequisites:

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10+
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU support)
- **Memory**: At least 8GB RAM (16GB recommended)
- **Storage**: 10GB of free disk space

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support recommended for training
- **CPU**: Multi-core processor (8+ cores recommended for large datasets)

## Installation Methods

### Option 1: Installation from Source (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vaibhavrathore1999/HiDISC.git
   cd HiDISC
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv hidisc_env
   source hidisc_env/bin/activate  # On Windows: hidisc_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install HIDISC in development mode**
   ```bash
   pip install -e .
   ```

### Option 2: Installation via PyPI (Coming Soon)

```bash
pip install hidisc
```

## Dependency Details

HIDISC relies on the following key dependencies:

### Core Dependencies
- **PyTorch**: 1.12.0+ with CUDA support
- **NumPy**: 1.21.0+
- **SciPy**: 1.7.0+
- **Scikit-learn**: 1.0.0+

### Hyperbolic Geometry
- **Geoopt**: For hyperbolic optimization
- **Hyperbolic**: Custom hyperbolic operations

### Data Processing
- **Torchvision**: 0.13.0+
- **PIL**: Image processing
- **OpenCV**: Computer vision operations

### Visualization
- **Matplotlib**: 3.5.0+
- **Seaborn**: Statistical plotting
- **Tensorboard**: Training visualization

## Verification

To verify your installation, run the following test:

```python
import hidisc
print(f"HIDISC version: {hidisc.__version__}")

# Test hyperbolic operations
from hidisc.geometry import HyperbolicSpace
h_space = HyperbolicSpace(dim=128)
print("Hyperbolic space initialized successfully!")
```

## GPU Setup

### CUDA Configuration

1. **Check CUDA availability**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

2. **Set CUDA device**
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Use first GPU
   ```

### Memory Optimization

For large datasets, consider:
- Using mixed precision training
- Gradient accumulation
- Model parallelism for multi-GPU setups

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size or use gradient accumulation
export BATCH_SIZE=16
export ACCUMULATE_GRAD_BATCHES=4
```

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

#### Permission Errors
```bash
# Use user installation
pip install --user -e .
```

### Getting Help

If you encounter issues during installation:

1. Check our [GitHub Issues](https://github.com/Vaibhavrathore1999/HiDISC/issues)
2. Create a new issue with:
   - Your operating system and Python version
   - Complete error messages
   - Steps to reproduce the problem

## Development Setup

For contributors and developers:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Docker Installation (Coming Soon)

For containerized deployment:

```bash
docker pull hidisc/hidisc:latest
docker run -it --gpus all hidisc/hidisc:latest
```

---

**Next Steps**: Once installation is complete, proceed to the [Usage Guide](usage.md) to start using HIDISC.