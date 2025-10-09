# Understanding FCAL Data

## Introduction

Before building machine learning models, we need to understand the structure, format, and characteristics of FCAL data. This chapter explores how FCAL data is organized and what features are available for analysis.

## Data Sources

### Simulation Data (Monte Carlo)

**HDGeant**: GlueX's Geant4-based simulation tool
- Generates particle interactions in the detector
- Provides "ground truth" labels for supervised learning
- Includes particle-level information (PDG codes, kinematics)

**Contents**:
- Generated events with known particle types
- Full detector response simulation
- Digitized signals matching real data format

**Advantages**:
- Perfect labels for training
- Unlimited statistics
- Control over event properties

**Limitations**:
- May not perfectly match real data
- Simulation systematics and uncertainties
- Computationally expensive to generate

### Real Data (Experimental)

**GlueX Runs**: Actual beam-on-target data
- Real detector conditions and noise
- Represents true physics to be analyzed
- No perfect labels (requires reconstruction)

**Use Cases**:
- Validation of trained models
- Transfer learning from simulation
- Understanding data-MC differences

## Data Format

### HDF5 Structure

FCAL data is typically stored in HDF5 format for efficient storage and access:

```python
import h5py
import numpy as np

# Open HDF5 file
with h5py.File('fcal_data.h5', 'r') as f:
    # List available datasets
    print("Available datasets:")
    for key in f.keys():
        print(f"  {key}: {f[key].shape}")
    
    # Load data
    images = f['images'][:]        # Shape: (N, H, W) or (N, C, H, W)
    labels = f['labels'][:]        # Shape: (N,)
    energies = f['energies'][:]    # Shape: (N,)
    positions = f['positions'][:]  # Shape: (N, 2)
```

Typical structure:
```
fcal_data.h5
├── images/            # FCAL shower images (N, 32, 32) or (N, 64, 64)
├── labels/            # Class labels: 0=photon, 1=splitoff
├── energies/          # Total cluster energy (GeV)
├── positions/         # Cluster (x, y) position (cm)
├── times/             # Cluster timing (ns)
├── particle_ids/      # PDG particle codes (simulation only)
├── momenta/           # Particle momentum (px, py, pz) (simulation only)
└── metadata/          # Run numbers, event numbers, etc.
```

### NumPy Arrays

Alternatively, data may be in NumPy format:

```python
# Load from .npy files
images = np.load('fcal_images.npy')
labels = np.load('fcal_labels.npy')

print(f"Images shape: {images.shape}")  # e.g., (10000, 32, 32)
print(f"Labels shape: {labels.shape}")  # e.g., (10000,)
```

### ROOT Files

Physics data often comes from ROOT files:

```python
import uproot
import awkward as ak

# Read ROOT file
file = uproot.open('fcal_data.root')
tree = file['fcal_tree']

# Access branches
energies = tree['cluster_energy'].array()
x_pos = tree['cluster_x'].array()
y_pos = tree['cluster_y'].array()
block_energies = tree['block_energies'].array()  # Jagged array

print(f"Number of events: {len(energies)}")
```

## Data Exploration

### Loading and Initial Inspection

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load data
with h5py.File('fcal_training_data.h5', 'r') as f:
    images = f['images'][:1000]  # Load first 1000 events
    labels = f['labels'][:1000]
    energies = f['energies'][:1000]

print(f"Dataset size: {len(images)} events")
print(f"Image shape: {images[0].shape}")
print(f"Energy range: {energies.min():.2f} - {energies.max():.2f} GeV")

# Class distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"\nClass distribution:")
for cls, count in zip(unique, counts):
    cls_name = 'photon' if cls == 0 else 'splitoff'
    print(f"  {cls_name}: {count} ({count/len(labels)*100:.1f}%)")
```

### Visualizing FCAL Showers

```python
def plot_fcal_showers(images, labels, energies, num_samples=8):
    """Plot sample FCAL shower images"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        idx = np.random.randint(len(images))
        img = images[idx]
        label = 'Photon' if labels[idx] == 0 else 'Splitoff'
        energy = energies[idx]
        
        ax = axes[i]
        im = ax.imshow(img, cmap='hot', origin='lower')
        ax.set_title(f'{label}, E={energy:.2f} GeV')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('fcal_shower_examples.png', dpi=150)
    plt.show()

plot_fcal_showers(images, labels, energies)
```

### Energy Distributions

```python
def plot_energy_distributions(energies, labels):
    """Compare energy distributions for different classes"""
    photon_energies = energies[labels == 0]
    splitoff_energies = energies[labels == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(photon_energies, bins=50, alpha=0.5, label='Photon', density=True)
    plt.hist(splitoff_energies, bins=50, alpha=0.5, label='Splitoff', density=True)
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Normalized Counts')
    plt.title('Energy Distributions by Particle Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('energy_distributions.png', dpi=150)
    plt.show()
    
    print(f"Photon energy: {photon_energies.mean():.2f} ± {photon_energies.std():.2f} GeV")
    print(f"Splitoff energy: {splitoff_energies.mean():.2f} ± {splitoff_energies.std():.2f} GeV")

plot_energy_distributions(energies, labels)
```

### Shower Shape Analysis

```python
def compute_shower_features(image):
    """Compute shower shape features"""
    # Energy-weighted centroid
    total_energy = image.sum()
    if total_energy == 0:
        return None
    
    y_coords, x_coords = np.meshgrid(range(image.shape[0]), range(image.shape[1]), indexing='ij')
    
    x_mean = (image * x_coords).sum() / total_energy
    y_mean = (image * y_coords).sum() / total_energy
    
    # RMS width
    x_rms = np.sqrt((image * (x_coords - x_mean)**2).sum() / total_energy)
    y_rms = np.sqrt((image * (y_coords - y_mean)**2).sum() / total_energy)
    
    # Number of blocks above threshold
    threshold = total_energy * 0.01  # 1% of total
    n_blocks = (image > threshold).sum()
    
    return {
        'total_energy': total_energy,
        'x_mean': x_mean,
        'y_mean': y_mean,
        'x_rms': x_rms,
        'y_rms': y_rms,
        'n_blocks': n_blocks
    }

# Compute features for all images
features = [compute_shower_features(img) for img in images]
features_df = pd.DataFrame([f for f in features if f is not None])

# Plot RMS distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(features_df[labels == 0]['x_rms'], bins=30, alpha=0.5, label='Photon')
plt.hist(features_df[labels == 1]['x_rms'], bins=30, alpha=0.5, label='Splitoff')
plt.xlabel('RMS Width (pixels)')
plt.ylabel('Counts')
plt.title('Transverse Shower Width')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(features_df[labels == 0]['n_blocks'], bins=20, alpha=0.5, label='Photon')
plt.hist(features_df[labels == 1]['n_blocks'], bins=20, alpha=0.5, label='Splitoff')
plt.xlabel('Number of Blocks')
plt.ylabel('Counts')
plt.title('Cluster Multiplicity')
plt.legend()

plt.tight_layout()
plt.savefig('shower_features.png', dpi=150)
plt.show()
```

## Data Quality Checks

### Checking for Issues

```python
def data_quality_checks(images, labels, energies):
    """Perform data quality checks"""
    print("Data Quality Report")
    print("=" * 50)
    
    # Check for NaNs or Infs
    nan_images = np.isnan(images).any(axis=(1, 2)).sum()
    inf_images = np.isinf(images).any(axis=(1, 2)).sum()
    print(f"Images with NaN: {nan_images}")
    print(f"Images with Inf: {inf_images}")
    
    # Check for zero-energy showers
    zero_energy = (images.sum(axis=(1, 2)) == 0).sum()
    print(f"Zero-energy showers: {zero_energy}")
    
    # Check energy consistency
    image_energies = images.sum(axis=(1, 2))
    energy_diff = np.abs(image_energies - energies)
    inconsistent = (energy_diff > 0.01 * energies).sum()
    print(f"Energy inconsistencies: {inconsistent}")
    
    # Check label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Check value ranges
    print(f"\nValue ranges:")
    print(f"  Image values: [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Energies: [{energies.min():.2f}, {energies.max():.2f}] GeV")
    
    # Statistical summary
    print(f"\nStatistical summary:")
    print(f"  Mean energy per event: {image_energies.mean():.2f} GeV")
    print(f"  Std energy per event: {image_energies.std():.2f} GeV")
    print(f"  Mean non-zero pixels: {(images > 0).sum(axis=(1,2)).mean():.1f}")

data_quality_checks(images, labels, energies)
```

### Handling Missing or Corrupt Data

```python
def clean_data(images, labels, energies):
    """Remove bad events from dataset"""
    # Find valid events
    valid_mask = (
        ~np.isnan(images).any(axis=(1, 2)) &  # No NaNs
        ~np.isinf(images).any(axis=(1, 2)) &  # No Infs
        (images.sum(axis=(1, 2)) > 0) &       # Non-zero energy
        (energies > 0)                         # Positive energy
    )
    
    n_removed = len(images) - valid_mask.sum()
    print(f"Removed {n_removed} bad events ({n_removed/len(images)*100:.2f}%)")
    
    return images[valid_mask], labels[valid_mask], energies[valid_mask]

images_clean, labels_clean, energies_clean = clean_data(images, labels, energies)
```

## Data Statistics

### Computing Dataset Statistics

```python
def compute_dataset_statistics(images, labels):
    """Compute mean and std for normalization"""
    # Separate by class
    photon_images = images[labels == 0]
    splitoff_images = images[labels == 1]
    
    stats = {
        'overall': {
            'mean': images.mean(),
            'std': images.std(),
            'min': images.min(),
            'max': images.max()
        },
        'photon': {
            'mean': photon_images.mean(),
            'std': photon_images.std(),
            'count': len(photon_images)
        },
        'splitoff': {
            'mean': splitoff_images.mean(),
            'std': splitoff_images.std(),
            'count': len(splitoff_images)
        }
    }
    
    return stats

stats = compute_dataset_statistics(images_clean, labels_clean)
print(f"Dataset Statistics:")
print(f"  Overall mean: {stats['overall']['mean']:.4f}")
print(f"  Overall std: {stats['overall']['std']:.4f}")
print(f"  Value range: [{stats['overall']['min']:.4f}, {stats['overall']['max']:.4f}]")

# Save statistics for later use
np.save('dataset_statistics.npy', stats)
```

## Multi-Particle Datasets

For generative models, we need data for multiple particle types:

```python
# Load data for different particle types
particle_types = {
    0: 'photon',
    1: 'pi_plus',
    2: 'pi_minus'
}

# Example structure
with h5py.File('fcal_multiparticle_data.h5', 'r') as f:
    all_images = f['images'][:]
    particle_ids = f['particle_ids'][:]
    energies = f['energies'][:]
    momenta = f['momenta'][:]  # (px, py, pz)
    
    # Separate by particle type
    for pid, name in particle_types.items():
        mask = (particle_ids == pid)
        n_events = mask.sum()
        print(f"{name}: {n_events} events")
        
        # Visualize a sample
        sample_idx = np.where(mask)[0][0]
        plt.figure(figsize=(6, 5))
        plt.imshow(all_images[sample_idx], cmap='hot')
        plt.title(f'{name} shower, E={energies[sample_idx]:.2f} GeV')
        plt.colorbar()
        plt.savefig(f'example_{name}_shower.png')
        plt.close()
```

## Summary

Key takeaways:
1. FCAL data is stored in HDF5, NumPy, or ROOT formats
2. Datasets include images, labels, energies, and kinematics
3. Data exploration reveals physics-motivated features
4. Quality checks ensure clean training data
5. Statistics inform preprocessing and normalization

## Next Steps

Now that you understand the data:
- Proceed to [Data Preprocessing](05-data-preprocessing.md) to prepare data for ML
- Jump to [Dataset Creation](06-dataset-creation.md) to build training/validation splits

## Further Reading

- HDF5 documentation: [hdfgroup.org](https://www.hdfgroup.org/)
- Uproot for ROOT files: [uproot.readthedocs.io](https://uproot.readthedocs.io/)
- Awkward Array for jagged data: [awkward-array.org](https://awkward-array.org/)
