# Quick Start Guide

Welcome to the DNP 2025 Deep Learning Tutorials! This guide will help you get started quickly.

## What You'll Learn

This tutorial teaches you to:
1. **Classify FCAL showers** using CNNs to distinguish photons from splitoffs
2. **Generate FCAL showers** using GANs for π⁺, π⁻, and photons
3. Apply deep learning best practices to experimental nuclear physics

## 5-Minute Start

### 1. Read the Overview
Start with the [README](README.md) to understand the project goals.

### 2. Set Up Your Environment
Follow [Setup Instructions](chapters/02-setup.md) to install dependencies:
```bash
# Clone repository
git clone https://github.com/ai4eic/DNP2025-tutorials.git
cd DNP2025-tutorials

# Create conda environment
conda create -n dnp2025 python=3.10
conda activate dnp2025

# Install dependencies
pip install -r requirements.txt
```

### 3. Explore the Structure
```
DNP2025-tutorials/
├── README.md              # Project overview
├── SUMMARY.md             # Table of contents
├── chapters/              # Tutorial chapters
│   ├── 01-overview.md     # Start here!
│   ├── 03-fcal-physics.md # Physics background
│   ├── 08-classification-model.md  # CNN tutorial
│   └── 12-gan-model.md    # GAN tutorial
├── notebooks/             # Jupyter notebooks (coming soon)
└── requirements.txt       # Python dependencies
```

### 4. Follow the Learning Path

**Beginner Path** (Start here):
1. [Tutorial Overview](chapters/01-overview.md)
2. [Setup and Installation](chapters/02-setup.md)
3. [FCAL Physics Background](chapters/03-fcal-physics.md)
4. [Understanding FCAL Data](chapters/04-data-understanding.md)

**Classification Path**:
5. [Building Classification Models](chapters/08-classification-model.md)
6. [References](chapters/19-references.md) for additional reading

**Generation Path**:
5. [GAN for FCAL Showers](chapters/12-gan-model.md)
6. [References](chapters/19-references.md) for additional reading

## Prerequisites

**Required:**
- Python 3.8+
- Basic Python programming
- Understanding of NumPy arrays
- Basic nuclear physics knowledge

**Recommended:**
- Machine learning concepts (can be learned alongside)
- PyTorch or TensorFlow experience (helpful but not required)

## Key Resources

### Documentation
- [README.md](README.md) - Project overview and goals
- [SUMMARY.md](SUMMARY.md) - Complete table of contents
- [BUILD.md](BUILD.md) - How to build/view the tutorial
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

### Tutorial Chapters (Fully Written)
- **[Chapter 1: Tutorial Overview](chapters/01-overview.md)** - Learning objectives and structure
- **[Chapter 2: Setup](chapters/02-setup.md)** - Installation and configuration
- **[Chapter 3: FCAL Physics](chapters/03-fcal-physics.md)** - Detector and physics background
- **[Chapter 4: Data Understanding](chapters/04-data-understanding.md)** - Data formats and exploration
- **[Chapter 8: Classification Models](chapters/08-classification-model.md)** - CNN architectures
- **[Chapter 12: GAN Models](chapters/12-gan-model.md)** - Generative models
- **[Chapter 19: References](chapters/19-references.md)** - Papers, books, and resources
- **[Chapter 20: Glossary](chapters/20-glossary.md)** - Term definitions

### Additional Chapters
- Chapters 5-7, 9-11, 13-18, 21 are placeholder chapters for future content

## Common Tasks

### View the Tutorial

**Option 1: Online (Recommended)**
- View the published Jupyter Book at [https://ai4eic.github.io/DNP2025-tutorials/](https://ai4eic.github.io/DNP2025-tutorials/)

**Option 2: GitHub** (Simple)
- Browse chapters directly on [GitHub](https://github.com/ai4eic/DNP2025-tutorials/tree/main/chapters)

**Option 3: Local Build**
- Build and view locally:
  ```bash
  jupyter-book build .
  open _build/html/index.html
  ```

See [BUILD.md](BUILD.md) for more options.

### Run Code Examples

```python
# Example: Simple CNN forward pass
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return x

# Test
model = SimpleCNN()
x = torch.randn(1, 1, 32, 32)
output = model(x)
print(f"Output shape: {output.shape}")
```

### Get Help

**Stuck?**
1. Check the [Glossary](chapters/20-glossary.md) for term definitions
2. Read the [FAQ](chapters/21-faq.md) (coming soon)
3. Open an [issue](https://github.com/ai4eic/DNP2025-tutorials/issues)
4. Refer to [References](chapters/19-references.md) for additional resources

## Next Steps

### Ready to Learn?

**For Classification:**
1. ✅ Complete setup (Chapter 2)
2. ✅ Read FCAL physics background (Chapter 3)
3. ✅ Understand data format (Chapter 4)
4. 📖 Build CNN classifier (Chapter 8)
5. 🔄 Train and evaluate (Chapters 9-10, coming soon)

**For Generation:**
1. ✅ Complete setup (Chapter 2)
2. ✅ Read FCAL physics background (Chapter 3)
3. ✅ Understand data format (Chapter 4)
4. 📖 Learn GAN basics (Chapters 11-12)
5. 🔄 Build generative models (Chapters 13-15, coming soon)

### Want to Contribute?

This is an open-source project! Contributions welcome:
- Fix typos or improve documentation
- Complete placeholder chapters
- Add example notebooks
- Share feedback

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Tutorial Status

**Completed:** ✅  
**In Progress:** 🔄  
**Planned:** 📋

| Chapter | Title | Status |
|---------|-------|--------|
| 01 | Tutorial Overview | ✅ |
| 02 | Setup and Installation | ✅ |
| 03 | FCAL Physics Background | ✅ |
| 04 | Understanding FCAL Data | ✅ |
| 05 | Data Preprocessing | 📋 |
| 06 | Dataset Creation | 📋 |
| 07 | Introduction to CNNs | 📋 |
| 08 | Building Classification Models | ✅ |
| 09 | Training and Optimization | 📋 |
| 10 | Model Evaluation | 📋 |
| 11 | Introduction to Generative Models | 📋 |
| 12 | GAN for FCAL Showers | ✅ |
| 13 | VAE for FCAL Showers | 📋 |
| 14 | Conditional Generation | 📋 |
| 15 | Physics-Informed Generation | 📋 |
| 16 | Transfer Learning | 📋 |
| 17 | Uncertainty Quantification | 📋 |
| 18 | Model Deployment | 📋 |
| 19 | References and Resources | ✅ |
| 20 | Glossary | ✅ |
| 21 | FAQ | 📋 |

## Support

**Questions?**
- 📧 Open an issue on GitHub
- 💬 Check the Glossary for terminology
- 📚 Refer to References for additional resources

**Found a Bug?**
- 🐛 Report it on [GitHub Issues](https://github.com/ai4eic/DNP2025-tutorials/issues)

**Want to Help?**
- 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Event Information:**  
DNP 2025 Tutorial Session  
Date: October 17, 2025  
Location: Chicago, IL

**Repository:** [github.com/ai4eic/DNP2025-tutorials](https://github.com/ai4eic/DNP2025-tutorials)

**License:** CC0 1.0 Universal (Public Domain)

---

*Ready? Start with [Chapter 1: Tutorial Overview](chapters/01-overview.md)! 🚀*
