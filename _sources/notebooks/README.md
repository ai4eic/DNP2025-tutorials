# Tutorial Notebooks

This directory will contain Jupyter notebooks for hands-on exercises and demonstrations.

## Planned Notebooks

### Data Exploration
- `01_data_exploration.ipynb` - Loading and visualizing FCAL data
- `02_data_preprocessing.ipynb` - Preprocessing and normalization
- `03_dataset_creation.ipynb` - Creating train/val/test splits

### Classification
- `04_simple_cnn.ipynb` - Building a simple CNN classifier
- `05_advanced_cnn.ipynb` - Advanced CNN architectures
- `06_model_training.ipynb` - Training and optimization
- `07_model_evaluation.ipynb` - Evaluation and metrics

### Generative Models
- `08_gan_basics.ipynb` - Introduction to GANs
- `09_conditional_gan.ipynb` - Conditional GAN for particle types
- `10_vae_model.ipynb` - Variational Autoencoder
- `11_physics_constrained.ipynb` - Physics-informed generation

### Advanced Topics
- `12_transfer_learning.ipynb` - Transfer learning examples
- `13_uncertainty_quantification.ipynb` - Uncertainty estimation
- `14_model_deployment.ipynb` - Model deployment examples

## How to Use

1. **Install Dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```

3. **Open Notebooks:**
   Navigate to the desired notebook and run cells sequentially.

## Data Requirements

Most notebooks require sample FCAL data. Download instructions:

```bash
# Example download command (actual URL will be provided)
wget https://example.com/fcal_tutorial_data.tar.gz
tar -xzf fcal_tutorial_data.tar.gz -C ../data/
```

Expected data structure:
```
DNP2025-tutorials/
├── data/
│   ├── fcal_photons_train.h5
│   ├── fcal_photons_test.h5
│   ├── fcal_splitoffs_train.h5
│   ├── fcal_splitoffs_test.h5
│   └── ...
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── ...
```

## Contributing

To contribute a notebook:

1. Follow the naming convention: `##_descriptive_name.ipynb`
2. Include:
   - Title and overview
   - Learning objectives
   - Required imports
   - Clear explanations
   - Working code cells
   - Visualizations
   - Exercises (optional)
   - Summary

3. Test thoroughly before submitting
4. See [CONTRIBUTING.md](../CONTRIBUTING.md) for details

## Notebook Template

A template notebook is provided: `notebook_template.ipynb`

Copy and customize for new tutorials:
```bash
cp notebook_template.ipynb your_new_notebook.ipynb
```

## Support

For questions or issues with notebooks:
- Check notebook-specific comments
- Refer to relevant chapter in tutorial
- Open an issue on GitHub
- See [FAQ](../chapters/21-faq.md)

---

*Notebooks are under development. Check back for updates!*
