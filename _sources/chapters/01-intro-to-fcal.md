# The Forward Calorimeter (FCAL) in GlueX

## Overview and Physics Role

The **Forward Calorimeter (FCAL)** is one of the two electromagnetic calorimeters of the GlueX spectrometer, located downstream of the target along the beamline.  
Its primary purpose is to **detect photons and electrons emitted at small polar angles** (from roughly 1°–12°) and to provide energy, position, and timing measurements necessary for event reconstruction and particle identification.  

Together with the **Barrel Calorimeter (BCAL)**, the FCAL ensures nearly hermetic photon coverage for GlueX, enabling precision reconstruction of neutral mesons such as $\pi^0$ and $\eta$, and facilitating photoproduction studies of resonances that decay via multiple photons.  

The FCAL plays a central role in:
- Detecting **forward-going photons** from $\pi^0$ or $\eta$ decays.
- Measuring **energy flow** in the forward region, crucial for event exclusivity.
- Providing fast **trigger primitives** for high-rate data acquisition.
- Enabling **electron identification** and **split-off rejection** in combination with tracking and timing systems.

---

## Detector Geometry and Design

### Overall Structure

The FCAL is a **lead-glass sampling calorimeter** consisting of  
**2800 individual modules** arranged in a roughly square matrix with a **beam hole** at the center to allow passage of the photon beam and low-angle particles.

Each module is made of **TF-1 lead-glass** blocks with dimensions:

$$
45\,\text{mm} \times 45\,\text{mm} \times 450\,\text{mm}
$$

corresponding to approximately **15 radiation lengths**.  
The blocks are wrapped in aluminized Mylar for optical isolation and read out at the back with **5 cm photomultiplier tubes (PMTs)**, typically **FEU-84-3** or **ETL 9214B**, coupled via optical grease.  

The matrix is built from **59 × 59** modules with a **4 × 4** hole at the center (beamline clearance).  
Modules are mounted on a precision steel support plate tilted slightly upward to maintain perpendicularity to the incoming photon trajectories.

```{figure} ../images/FCAL_photo.png
---
alt: Forward Calorimeter geometry
width: 80%
---
:caption: Layout of the GlueX Forward Calorimeter showing the 59×59 module arrangement and central beam hole.
```

### Readout and Electronics

Each FCAL module is instrumented with a **photomultiplier tube (PMT)** coupled directly to the back face of the lead-glass block using optical grease.  
The PMTs (primarily ETL 9214B or Russian FEU-84-3 models) operate with custom-built voltage divider bases that ensure linear gain up to 10⁴ photoelectrons.  

Signals are transmitted via shielded cables to **12-bit 250 MHz flash ADC (fADC250)** modules housed in VXS crates.  
The fADCs continuously digitize the waveform, storing it in a **pipeline buffer** that allows for trigger latencies up to ≈ 3.3 µs.  

At the hardware level, the electronics system provides:
- Dynamic range from a few MeV up to several GeV,
- Timing resolution better than 2 ns per channel,
- Summed analog signals for **trigger formation** (4×4 block sums),
- Zero-suppression and pedestal subtraction performed online.

Digitized hits are read out through the DAQ and converted into calibrated ADC counts during reconstruction.

---

## Energy Measurement and Resolution

The FCAL provides both **energy** and **timing** measurements for electromagnetic showers.  
Calibration constants convert digitized pulse integrals into physical energy values.

### Energy Reconstruction

The calibrated energy in each channel is computed as:

$$
E_i = C_i \times (Q_i - P_i),
$$

where:
- \( Q_i \) is the integrated ADC value,
- \( P_i \) is the pedestal (baseline),
- \( C_i \) is a calibration coefficient (MeV/ADC count).

Cluster energy is obtained as the sum over contributing cells:

$$
E_{\text{cluster}} = \sum_{i \in \text{cluster}} E_i.
$$

---

### Energy Resolution

From beam and simulation studies, the FCAL energy resolution follows:

$$
\frac{\sigma_E}{E} = \frac{5.7\%}{\sqrt{E \, (\text{GeV})}} \oplus 1.0\%.
$$

The first term arises from shower statistics and photoelectron yield, while the constant term accounts for calibration uncertainties and optical effects.

Linearity is maintained up to at least 6 GeV with less than 2% deviation from expectation.

---

### Position and Timing Resolution

The shower centroid is computed as an energy-weighted mean:

$$
x_c = \frac{\sum_i x_i E_i}{\sum_i E_i}, \qquad
y_c = \frac{\sum_i y_i E_i}{\sum_i E_i}.
$$

Typical resolutions:
- **Position:** 5 mm / √E(GeV)
- **Timing:** 150–200 ps at 1 GeV

Precise timing allows discrimination between prompt and accidental photons in the forward region (upto $2~ns$).

