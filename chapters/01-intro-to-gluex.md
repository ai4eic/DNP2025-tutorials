# Introduction to GlueX

## Scientific Motivation

The **GlueX** experiment is a dedicated photoproduction experiment at Jefferson Lab (Hall D), designed to explore the spectrum of light-quark mesons, with special sensitivity to **hybrid and exotic mesons** that carry gluonic excitations beyond the simple quark–antiquark picture. {cite}`GlueX-overview, Zarling-RecentResults`  

Lattice QCD and phenomenological models predict that **hybrid mesons** with exotic quantum numbers (for example $J^{PC} = 1^{-+}, 0^{+-}, 2^{+-}$) should exist in the light-quark sector. These states are difficult to access in many production modes but may be favorable in photoproduction at intermediate energies. {cite}`Swanson-ExoticReview, GlueX-overview, Zarling-RecentResults`  

GlueX aims to provide a high-statistics, high-acceptance data set with linearly polarized photons to enable precision spectroscopy and amplitude analyses. {cite}`GlueX-overview, Zarling-RecentResults`  

---

## Experimental Apparatus and Beamline

### CEBAF and Hall D

The Continuous Electron Beam Accelerator Facility (CEBAF) at JLab delivers up to 12 GeV electron beams, which feed into multiple experimental halls. The Hall D complex was added as part of the 12 GeV upgrade to house the GlueX experiment. {cite}`GlueX-overview, NIM-A-GlueX`  

### Photon Beam Production and Tagging

To create a **linearly polarized photon beam** near 9 GeV, GlueX uses *coherent bremsstrahlung* from a thin diamond radiator. Electrons that emit photons lose energy and are bent more strongly by downstream dipoles; their position in the tagging spectrometer determines the photon energy. The polarization is maximized near a coherent peak in the spectrum. {cite}`NIM-A-GlueX, GlueX-overview`  

Photon flux is monitored via a **pair spectrometer**, and polarization is measured by a **triplet polarimeter**. {cite}`NIM-A-GlueX`  

### Target and Trigger

The photon beam interacts with a **liquid hydrogen target** (30 cm long) inside the spectrometer. Downstream detectors form triggers based on multiplicities, energy deposits, and timing. The DAQ system supports high trigger rates (tens of kHz) and uses pipelined electronics (flash ADCs, TDCs) with ~3.3 µs latency. {cite}`NIM-A-GlueX`  

---

## Detector Systems Overview

Below is a simplified schematic of the GlueX detector in Hall D:

```{figure} ../images/gluex_detector_schematic.png
---
alt: GlueX detector schematic
width: 80%
---
Cut-away view of GlueX in Hall D (not to scale).  
```

The detector is azimuthally symmetric and nearly hermetic for charged and neutral particles. {cite}`NIM-A-GlueX`

Major subsystems:

* **Tracking detectors**
    - Central Drift Chamber (CDC / straw tubes)
    - Forward Drift Chambers (FDC) (planar wire & cathode strips)
* **Time of Flight (TOF) system**
* **Start counter (SC)**
* **Barrel Calorimeter (BCAL)** -- The Barrel Calorimeter (BCAL), built with layers of scintillating fibers and lead, surrounds the tracking region. Its design allows coverage for photons over a large polar angle range. It was commissioned in 2014 and operates routinely. Performance includes: $ \frac{\sigma_E}{E} \approx \frac{5.2\%}{\sqrt{E}} \oplus 3.6\% $, timing resolution ~150 ps at 1 GeV. {cite}`Beattie-BCAL`
* **Forward Calorimeter** -- The forward calorimeter uses lead-glass modules arranged in a matrix to detect forward-going photons. (We will use FCAL data in later tutorial sections.)
* **Particle Identification** -- In early GlueX running (Phase I) the PID was based on $\frac{dE}{dx}$ and 
**TOF**. In the Phase II program, a DIRC (Detection of Internally Reflected Cherenkov light) system (recycled from BaBar) was installed to enhance kaon/pion ($K/\pi$) discrimination, especially in the forward direction {cite}`Beattie-BCAL, DIRC-commissioning`.


## Data Taking, Simulation, and Reconstruction Software

### Run Periods and Beam Conditions

GlueX began commissioning in 2014, then gradually ramped to physics-quality data in 2016 and beyond. The approved Phase I program ran through 2018, collecting large data sets. Phase II is ongoing, featuring the DIRC upgrade and higher photon fluxes. {cite}`GlueX-overview, Future-GlueX`

Photon flux in Phase I ranged from $1\times10^{7}$ to $3\times 10^{7} \gamma/s$ in the coherent peak. {cite}`Future-GlueX`

### Simulation Framework

GlueX uses a GEANT4-based simulation (HDGeant4) to model detector response, geometry, and materials. Digitization and smearing steps generate simulated detector hits. Conditions (e.g. calibration constants) are pulled from databases (CCDB, RCDB). {cite}`NIM-A-GlueX, GlueXWiki`


### Reconstruction and Analysis Workflow

The data and MC are processed through `halld_recon`, using `JANA2` (`JANA`) as the event engine. Reconstruction includes:

* Hit clustering, track finding, and fitting
* Shower and cluster association
* Kinematic fitting, vertexing, and particle ID
* Skim and reduction to analysis-level ntuples
* Reaction factories / amplitude analysis

The JANA/DANA framework enables modular plugins for physics skims and format conversion.

```{figure} ../images/simreconFlow.png
---
alt: GlueX detector schematic
width: 80%
---
Flow chart on simulation and reconstruction pipeline used at GlueX.  
```

## References

```{bibliography}
:style: unsrt
```