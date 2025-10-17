# Problem Formulation

The Forward Calorimeter (FCAL) in the GlueX detector is a high-granularity electromagnetic calorimeter composed of lead-glass modules arranged in a square grid. It measures the position and energy of electromagnetic (EM) showers produced by incident photons and electrons. When a high-energy photon enters the FCAL, it converts into an $e^+e^-$ pair, initiating a cascade of secondary particles that deposit energy across neighboring calorimeter cells. The resulting two-dimensional energy map — the **hit pattern** — reflects the spatial development of the shower.

In this study, we focus exclusively on **photon-induced showers** simulated with the GlueX Monte Carlo (HDGeant4 + halld_recon). From these reconstructed showers, we extract **localized patches** of the FCAL energy map centered around the shower maximum. Each patch encodes a localized shower image in the $(\text{row}, \text{col})$ grid, normalized to the total deposited energy.

---

## Problem Definition

The goal of this exercise is to develop a **Generative Adversarial Network (GAN)** that can **synthesize realistic FCAL shower images**, conditioned on the **thrown photon energy ($E_{\text{thrown}}$)**.

Formally, the generative model learns a conditional mapping:

$$
G(z, E_{\text{thrown}}) \rightarrow \hat{I}_{\text{shower}}
$$

where:
- $z$ is a random latent vector,
- $E_{\text{thrown}}$ is the conditioning variable (photon energy),
- $\hat{I}_{\text{shower}}$ is the generated shower image (energy pattern).

The discriminator is trained to distinguish **real** shower patches (from simulation) from **fake** ones (produced by the generator), also conditioned on $E_{\text{thrown}}$.

This conditional GAN (cGAN) framework enables the model to learn how the **shape and spread of the shower** evolve with photon energy — for instance, higher-energy photons producing deeper and broader cascades.

---

## Dataset and Preprocessing

The dataset is built from reconstructed **photon showers** only. We will use the same dataset used for [classification](../notebooks/02-cnn-classification.ipynb) and [regression](../notebooks/02-cnn-regression.ipynb):

1. **Event Selection**  
   - Only neutral showers with true particle ID = γ are included.  
   - Showers must be within the FCAL fiducial region to avoid edge effects.

2. **Patch Extraction**  
   - For each shower, an $N \times N$ patch (e.g., $11 \times 11$) centered on the cell with maximum deposited energy is extracted.  
   - The patch represents the local spatial distribution of deposited energy.

3. **Normalization**  
   - Energy values are normalized using a log-scaling transformation to compress the dynamic range:
     $$
     E' = \frac{\log(1 + E / E_0)}{\log(1 + E_{\text{max}} / E_0)}
     $$
   - This ensures numerical stability and highlights relative shower shapes rather than absolute scales.

4. **Condition Variable**  
   - The thrown photon energy ($E_{\text{thrown}}$) is recorded for each patch and normalized to [0, 1].  
   - This is used as the conditioning input to both generator and discriminator.

The dataset can be found in [huggingface](https://huggingface.co/datasets/AI4EIC/DNP2025-tutorial/resolve/main/formatted_dataset/CNN4FCAL_GUN_PATCHSIZE_11.h5).



```{figure} ../images/true_patches.png
---
alt: GAN training data
width: 80%
---
:caption: A sample training dataset to generate FCAL shower energy patches is shown below. The pixel energies and thrownE are normalized as mentioned above.
```


## Data preprocessing

For this demonstration, the dataset is restricted to events with **thrown photon energies between 1.0 GeV and 2.0 GeV**.

This limited range serves two purposes:

1. **Computational Feasibility:**  
   Restricting to a narrow energy window ensures that the GAN can be trained efficiently — even on CPU — within a reasonable number of epochs.  
   It also reduces the overall dynamic range of shower intensities, simplifying the conditional learning task.

2. **Controlled Conditioning:**  
   By fixing a limited energy range, the model focuses on learning **fine-grained variations in shower shape** rather than global scaling with energy.  
   This allows clearer interpretation of the conditioning effect and better visual inspection of generated patterns.

### Hit Energy Threshold

Because a GAN learns to generate *physically plausible* but not *pixel-by-pixel identical* hit patterns, a small energy threshold is applied to define realistic cell occupancies.  
In particular, we apply a **per-cell energy cut of 0.05 GeV (50 MeV)** when evaluating or visualizing generated showers:

$$
E_{\text{cell}} \ge 0.05 \, \text{GeV}
$$

This threshold reflects the approximate detection sensitivity of the Forward Calorimeter and helps eliminate numerical noise or unphysical low-level fluctuations in the generated images.

By doing so, the comparison between real and generated showers focuses on meaningful energy deposits — i.e., those that would actually produce measurable detector responses — rather than on statistical pixel noise or reconstruction artifacts.

### Practical Implication

With these cuts:
- The GAN trains faster and more stably due to reduced input variance.  
- Visual and quantitative metrics (e.g., occupancy, E1/E9) become more physically interpretable.  
- Generated showers correspond to realistic FCAL responses within the 1–2 GeV photon energy regime, making them ideal for illustrating the principles of **conditional generative modeling in calorimeter simulation**.


```{note}
Model trained on the full range with $0.01 \leq \text{thrownE} \leq 4.0~[\text{GeV}]$ with a minimum Hit Energy threshold of $0.01$[GeV] can be found in [`generatorGAN-FCAL-100MeV-4GeV.safetensors` ![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue.svg?logo=huggingface)](https://huggingface.co/AI4EIC/DNP2025-tutorial/resolve/main/generatorGAN-FCAL-100MeV-4GeV.safetensors) 
```