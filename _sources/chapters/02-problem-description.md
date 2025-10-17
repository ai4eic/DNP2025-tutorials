# Problem Formulation

The Forward Calorimeter (FCAL) in the GlueX detector is an electromagnetic (EM) calorimeter located downstream of the target. It is designed primarily to measure the energies and positions of photons and electrons by absorbing their energy in dense lead-glass blocks. When a photon or electron enters the FCAL, it initiates an electromagnetic shower — a cascade of secondary photons, electrons, and positrons — that deposit energy across several adjacent calorimeter cells.

In contrast, hadronic particles (such as $\pi^{+}$, $\pi^{-}$, protons, or neutrons) interact via both electromagnetic and strong interactions, leading to more complex and less localized energy deposition patterns. These interactions can produce **hadronic split-offs**, i.e., secondary interactions in which a single charged hadron produces additional neutral secondaries (such as $\pi^{0}$ or $\gamma$) that create **separate electromagnetic-like showers** in the calorimeter.

A single charged particle traversing the detector can therefore give rise to **multiple reconstructed showers**:
- One shower typically associated with the charged track (track-matched shower).
- Additional showers, arising from split-offs, that are not associated with any charged track.

By convention, **showers that are not matched to a reconstructed track** are classified as *neutral showers*. These include:
- **True photon showers**, originating from neutral particles in the event (e.g., from π⁰ → γγ decays).
- **Split-off showers**, arising from secondary hadronic interactions.

### Foundational Work and Context

The foundational work relevant to this tutorial was laid by the work of **Rebecca Barsotti and Matt Shepherd**, whose analysis and methodology in identifying and classifying FCAL showers provided a systematic framework for distinguishing true photon showers from hadronic split-offs in GlueX data {cite}`Barsotti-2020`.  

Their study established the critical role of shower-shape variables, spatial correlations, and topological information in separating these categories. The insights gained from that work form the **conceptual and methodological basis** for this tutorial

## Problem Definition

The key challenge addressed here is to **discriminate true photon showers from hadronic split-off showers** in the FCAL.

Given the *hit-level information* of a reconstructed neutral shower — specifically the pattern of energy deposits across FCAL cells (rows, columns, and deposited energies) — we seek to train a **Convolutional Neural Network (CNN)** that can classify each neutral shower as either:

- **Photon** (good EM shower), or  
- **Split-off** (hadronic contamination).

Formally, the task can be defined as a binary classification problem:

$
f(\text{hit map}) \rightarrow \{0, 1\}
$

where `0` denotes *split-off* and `1` denotes *photon*.

---

## Pipeline and Dataset Creation

The data preparation pipeline consists of two main stages:

1. **Training Dataset (Particle Gun Sample)**  
   Generate single-particle events (γ, π⁺, π⁻) using the GlueX Monte Carlo simulation chain. Each event produces one or more FCAL showers, which are labeled according to the particle type and whether the shower is matched to a track or not. These labeled samples are used to train the CNN classifier.

2. **Testing Dataset (Physics Reaction: ω → π⁺ π⁻ π⁰)**  
   Generate exclusive ω production events where π⁰ decays into two photons. The CNN trained on the particle-gun dataset is then applied to identify good photon showers in these realistic multi-particle events.

---

### Particle Gun Sample

To generate a clean and controlled dataset, single-particle events are produced using the GlueX *particle gun* generators. The generated particles are uniformly sampled within specific kinematic ranges, ensuring coverage of the FCAL acceptance (1°–11° in polar angle).

The generated kinematic configuration is summarized below:

| Particle Name | Momentum Range [GeV] | $\theta$ Range [deg] | $\phi$ Range [deg] | No. of Events |
|----------------|----------------------|----------------|----------------|----------------|
| $\gamma$ (Photon)     | 0.1 – 4.0            | 1 – 11         | 0 – 360        | 500k           |
| $\pi^{+}$             | 0.1 – 4.0            | 1 – 11         | 0 – 360        | 250k           |
| $\pi^{-}$             | 0.1 – 4.0            | 1 – 11         | 0 – 360        | 250k           |

Each event is simulated through the full GlueX detector response (HDGeant4), and reconstructed using the standard `halld_recon` pipeline.  
From the reconstructed data:
- Showers **matched to charged tracks** are labeled as *hadronic showers*.
- Neutral showers (no track match) are further labeled as *photon* or *split-off*, depending on the thrown particle and reconstruction.

This dataset provides the supervised labels needed to train the CNN on spatial energy patterns (row, column, energy).

---

### Omega Exclusive Sample

For realistic evaluation, we use the exclusive $\omega$ production process:

$
\gamma p \rightarrow p \, \omega \rightarrow p \, (\pi^+ \pi^- \pi^0) \rightarrow p \, (\pi^+ \pi^- \gamma_1 \gamma_2)
$

This sample is generated using the **`gen_omega_3pi`** event generator.  
Kinematic conditions are chosen to ensure that at least one of the charged pions ($\pi^{+}$ or $\pi^{-}$) and at least one photon from $\pi^{0}$ decay are within the FCAL acceptance.

Event-level selection criteria:
- Events must have **exactly three reconstructed charged tracks** corresponding to p, $\pi^{+}$, and $\pi^{-}$.
- At least two **neutral showers** reconstructed in the FCAL.
- Reconstructed track momenta are assumed to be well-measured (perfect tracking approximation).

In real reconstructed $\omega$ events, the FCAL typically contains **more than two neutral showers** due to hadronic split-offs or secondary interactions.  
The CNN is therefore used to classify each neutral shower as *good photon* or *split-off*.

### CNN Behavior Consideration

Since the CNN operates independently on each shower:
- It may **predict more than two** good photon showers per event, or **fewer than two**.
- This behavior must be handled during event reconstruction by selecting the **two highest-confidence photon predictions** for subsequent $\pi^{0}$ and $\omega$ reconstruction.

This step ensures consistent physics reconstruction downstream, while still evaluating the network’s discriminative power on realistic, noisy data.

---

## Summary of the Dataset Pipeline

1. **Simulate particle-gun data** for $\gamma$, $\pi^{+}$, and $\pi^{-}$ across FCAL acceptance.  
   → Label and preprocess into shower images for CNN training.

2. **Train CNN** to classify neutral showers as photons or split-offs.

3. **Generate ω → π⁺ π⁻ π⁰ → π⁺ π⁻ γγ events**.  
   → Apply the trained CNN to select photon candidates in realistic multi-shower environments.

4. **Evaluate performance** using reconstruction metrics such as:
   - Correct photon identification rate per event
   - $\pi^{0}$ and $\omega$ invariant mass resolution
   - False positive rate (split-offs misidentified as photons)

This integrated pipeline connects detector-level physics simulation with a machine learning surrogate, providing both a pedagogical and practical framework for accelerating simulation and reconstruction tasks in nuclear physics.


## A sample Event

A typical event in FCAL will look like this. One can treat the FCAL readout as an image. 

```{figure} ../images/FCALEvent.png
---
alt: Forward Calorimeter event
width: 80%
---
:caption: This is a typical FCAL Event. Colored blocks show cells registering an energy deposition. Circles indicate showers identified by the reconstruction algorithm, where ther adius of the circle is proportional to the energy of the shower. Different colors represent various candidates (photon, charged or splitoffs). The figure is taken from {cite}`Barsotti-2020`
```

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```